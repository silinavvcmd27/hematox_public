# Выгрузка из Seurat: координаты клеток, тип клетки, ПОДТИП и экспрессия.
#
# ЧТО ИЗМЕНИЛОСЬ по сравнению с прежней версией:
#   1. Дополнительно пишется <slide>_expr.csv с экспрессией маркерных генов —
#      без него нельзя разделить строму на гормональную и матриксную.
#   2. Выгружается ещё и ПОДТИП клетки (cell_subtype). Это главное. В
#      Annotation_main_types_new все фибробласты слиты в "Fibroblasts"/"CAFs",
#      а стероидогенные популяции (STAR+ C7+ CAFs, CYP11A1+ fibroblasts,
#      Granulosa cells) видны только на уровне подтипов. Без них гормональную
#      строму не разметить.
#   3. Пишется <slide>_subtype_expr.csv — средняя экспрессия маркеров по
#      каждому подтипу (псевдобалк). Файл крошечный, именно по нему
#      принимается решение, какой подтип куда отнести.
#   4. Координаты берутся строго по именам x_centroid/y_centroid, а не
#      "первые два столбца, если не нашли" — молчаливый откат на столбцы 1 и 2
#      мог перепутать оси и сдвинуть всю разметку.
#   5. Файл записывается атомарно (через временное имя), чтобы прерванная
#      выгрузка не оставила обрезанный CSV, который потом молча прочитается.
#
# Запуск:
#   Rscript R/export_seurat.R /mnt/.../seurat2_ann_all.rds
#
# Имя среза берётся из имени файла. Чтобы задать своё имя и/или свою колонку
# подтипов, пишите через двоеточие:
#   Rscript R/export_seurat.R ovary2_he:Annotation_3_iter_new=/mnt/.../seurat_3_iteration.rds

suppressPackageStartupMessages(library(Seurat))

out_dir <- "data/seurat_csv"
celltype_col <- "Annotation_main_types_new"

# Колонка с подтипами. Ищется по порядку — берётся первая существующая.
# В ваших трёх срезах она называется по-разному, поэтому список, а не имя.
SUBTYPE_COLS <- c("Annotation_all_types_new",  # ovary_prime
                  "Annotation_3_iter_new",     # ovary2
                  "Annotation_2_iter",         # ovary3
                  "All_types_3", "Annotation_common", "Fibr_ann_only")

# Гены для разделения стромы. Стероидогенез — гормон-продуцирующая строма,
# матрикс — десмопластическая. Отсутствующие в панели Xenium просто
# пропускаются с предупреждением.
MARKERS <- c(
  # стероидогенез
  "STAR", "CYP11A1", "CYP17A1", "CYP19A1", "HSD3B2", "HSD17B1",
  "FOXL2", "NR5A1", "INHA", "INHBA", "AMH", "GATA4",
  # матрикс
  "COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL6A3", "FAP", "POSTN",
  "THY1", "LUM", "DCN", "FN1", "SPARC", "TAGLN", "ACTA2", "PDGFRB",
  "MMP2", "MMP11", "TIMP1"
)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) stop("передай .rds файлы аргументами")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
all_types <- character(0)
all_subs <- character(0)

write_atomic <- function(df, path) {
  tmp <- paste0(path, ".tmp")
  write.csv(df, tmp, row.names = FALSE)
  file.rename(tmp, path)
}

get_celltype <- function(obj) {
  md <- obj@meta.data
  if (celltype_col %in% colnames(md)) return(as.character(md[[celltype_col]]))
  warning("колонки '", celltype_col, "' нет, беру Idents()", call. = FALSE)
  as.character(Idents(obj))
}

# Подтип: первая найденная колонка из SUBTYPE_COLS, либо заданная явно.
# Печатает все подходящие колонки с числом уровней, чтобы было видно, из чего
# выбиралось, и не пришлось гадать.
get_subtype <- function(obj, forced = NA) {
  md <- obj@meta.data
  cand <- intersect(SUBTYPE_COLS, colnames(md))
  if (length(cand)) {
    message("  колонки подтипов в объекте:")
    for (cc in cand)
      message("    ", cc, " — уровней: ", length(unique(md[[cc]])))
  }
  col <- NA
  if (!is.na(forced)) {
    if (!(forced %in% colnames(md)))
      stop("колонки '", forced, "' нет в объекте. Есть: ",
           paste(cand, collapse = ", "))
    col <- forced
  } else if (length(cand)) {
    col <- cand[1]
  }
  if (is.na(col)) {
    warning("колонки подтипов не нашла, cell_subtype = cell_type", call. = FALSE)
    return(list(v = NULL, col = NA))
  }
  message("  подтипы беру из: ", col)
  list(v = as.character(md[[col]]), col = col)
}

get_coords <- function(obj) {
  xy <- as.data.frame(GetTissueCoordinates(obj))
  cn <- tolower(colnames(xy))
  xi <- which(cn %in% c("x_centroid", "x", "imagecol", "col"))[1]
  yi <- which(cn %in% c("y_centroid", "y", "imagerow", "row"))[1]
  if (is.na(xi) || is.na(yi)) {
    stop("не нашла столбцы координат в GetTissueCoordinates(); есть: ",
         paste(colnames(xy), collapse = ", "),
         "\nдобавь нужное имя в get_coords(), не полагайся на порядок столбцов")
  }
  data.frame(x = as.numeric(xy[[xi]]), y = as.numeric(xy[[yi]]))
}

# "имя:колонка=путь", "имя=путь" или просто "путь"
parse_arg <- function(a) {
  slide <- NA; subcol <- NA; path <- a
  if (grepl("=", a, fixed = TRUE)) {
    lhs <- sub("=.*$", "", a)
    path <- sub("^[^=]*=", "", a)
    if (grepl(":", lhs, fixed = TRUE)) {
      slide <- sub(":.*$", "", lhs); subcol <- sub("^[^:]*:", "", lhs)
    } else slide <- lhs
  }
  if (is.na(slide)) slide <- tools::file_path_sans_ext(basename(path))
  list(slide = slide, subcol = subcol, path = path)
}

for (a in args) {
  p <- parse_arg(a)
  path <- p$path; slide <- p$slide
  message("читаю ", path)
  if (!file.exists(path)) stop("файла нет: ", path)
  obj <- readRDS(path)
  if (inherits(try(validObject(obj), silent = TRUE), "try-error"))
    obj <- UpdateSeuratObject(obj)

  ct <- get_celltype(obj)
  sub <- get_subtype(obj, p$subcol)
  xy <- get_coords(obj)
  ids <- colnames(obj)
  n <- min(nrow(xy), length(ct), length(ids))
  if (!is.null(sub$v)) n <- min(n, length(sub$v))
  if (n < length(ids)) {
    warning("длины не совпали: клеток ", length(ids), ", координат ", nrow(xy),
            ", аннотаций ", length(ct), " — беру первые ", n, call. = FALSE)
  }

  st <- if (is.null(sub$v)) ct[seq_len(n)] else sub$v[seq_len(n)]
  cells <- data.frame(cell_id = ids[seq_len(n)],
                      x = xy$x[seq_len(n)], y = xy$y[seq_len(n)],
                      cell_type = ct[seq_len(n)],
                      cell_subtype = st)
  write_atomic(cells, file.path(out_dir, paste0(slide, "_cells.csv")))
  message("  клетки -> ", nrow(cells), " строк, подтипов: ",
          length(unique(cells$cell_subtype)))

  # --- экспрессия маркеров ---
  present <- intersect(MARKERS, rownames(obj))
  absent <- setdiff(MARKERS, rownames(obj))
  if (length(absent)) {
    message("  нет в панели (", length(absent), "): ",
            paste(absent, collapse = ", "))
  }
  if (length(present) == 0) {
    warning("ни одного маркерного гена нет в объекте — файл экспрессии не пишу",
            call. = FALSE)
  } else {
    # counts, а не data: нормировку делает check_stroma_split.py, чтобы
    # она была одинаковой для всех срезов
    m <- GetAssayData(obj, layer = "counts")[present, seq_len(n), drop = FALSE]
    e <- cbind(data.frame(cell_id = ids[seq_len(n)]),
               as.data.frame(as.matrix(Matrix::t(m))))
    write_atomic(e, file.path(out_dir, paste0(slide, "_expr.csv")))
    message("  экспрессия -> ", length(present), " генов x ", nrow(e), " клеток")

    # Псевдобалк по подтипам: средняя экспрессия каждого маркера в каждом
    # подтипе + число клеток. По этой таблице решается, какой подтип
    # гормональный, а какой матриксный. Нормировка на глубину: доля от суммы
    # всех транскриптов клетки, потом log1p — иначе крупные клетки дают
    # больший балл просто из-за размера.
    tot <- Matrix::colSums(GetAssayData(obj, layer = "counts")[, seq_len(n),
                                                               drop = FALSE])
    tot[tot == 0] <- 1
    norm <- log1p(Matrix::t(m) / tot * 1e3)
    agg <- aggregate(as.matrix(norm), by = list(cell_subtype = st), FUN = mean)
    agg$n_cells <- as.integer(table(st)[agg$cell_subtype])
    agg$cell_type <- vapply(agg$cell_subtype, function(z)
      names(sort(table(cells$cell_type[st == z]), decreasing = TRUE))[1], "")
    agg <- agg[, c("cell_subtype", "cell_type", "n_cells",
                   setdiff(colnames(agg), c("cell_subtype", "cell_type",
                                            "n_cells")))]
    write_atomic(agg, file.path(out_dir, paste0(slide, "_subtype_expr.csv")))
    message("  псевдобалк -> ", nrow(agg), " подтипов")
  }

  all_types <- union(all_types, unique(cells$cell_type))
  all_subs <- union(all_subs, unique(cells$cell_subtype))
}

message("\nвсе cell_type (", length(all_types), "):")
for (t in sort(all_types)) cat("  -", t, "\n")
writeLines(sort(all_types), file.path(out_dir, "all_cell_types.txt"))
writeLines(sort(all_subs), file.path(out_dir, "all_cell_subtypes.txt"))
message("подтипов всего: ", length(all_subs),
        " -> data/seurat_csv/all_cell_subtypes.txt")
message("\nДальше: python check_stroma_split.py --subtype-expr ",
        "data/seurat_csv/*_subtype_expr.csv")
