# Выгрузка из Seurat: координаты клеток, тип клетки И экспрессия маркерных генов.
#
# ЧТО ИЗМЕНИЛОСЬ по сравнению с прежней версией:
#   1. Дополнительно пишется <slide>_expr.csv с экспрессией маркерных генов —
#      без него нельзя разделить строму на гормональную и матриксную.
#   2. Координаты берутся строго по именам x_centroid/y_centroid, а не
#      "первые два столбца, если не нашли" — молчаливый откат на столбцы 1 и 2
#      мог перепутать оси и сдвинуть всю разметку.
#   3. Файл записывается атомарно (через временное имя), чтобы прерванная
#      выгрузка не оставила обрезанный CSV, который потом молча прочитается.
#
# Запуск:
#   Rscript R/export_seurat.R data/seurat/ovary2.rds data/seurat/ovary3.rds

suppressPackageStartupMessages(library(Seurat))

out_dir <- "data/seurat_csv"
celltype_col <- "Annotation_main_types_new"

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

for (path in args) {
  message("читаю ", path)
  obj <- readRDS(path)
  slide <- tools::file_path_sans_ext(basename(path))

  ct <- get_celltype(obj)
  xy <- get_coords(obj)
  ids <- colnames(obj)
  n <- min(nrow(xy), length(ct), length(ids))
  if (n < length(ids)) {
    warning("длины не совпали: клеток ", length(ids), ", координат ", nrow(xy),
            ", аннотаций ", length(ct), " — беру первые ", n, call. = FALSE)
  }

  cells <- data.frame(cell_id = ids[seq_len(n)],
                      x = xy$x[seq_len(n)], y = xy$y[seq_len(n)],
                      cell_type = ct[seq_len(n)])
  write_atomic(cells, file.path(out_dir, paste0(slide, "_cells.csv")))
  message("  клетки -> ", nrow(cells), " строк")

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
  }

  all_types <- union(all_types, unique(cells$cell_type))
}

message("\nвсе cell_type (", length(all_types), "), разнеси их по группам ",
        "в config/cell_type_map.yaml:")
for (t in sort(all_types)) cat("  -", t, "\n")
writeLines(sort(all_types), file.path(out_dir, "all_cell_types.txt"))
