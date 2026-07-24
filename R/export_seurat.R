# Выгрузка координат клеток + cell_type из Seurat-объектов в CSV.

suppressPackageStartupMessages(library(Seurat))

out_dir <- "data/seurat_csv"
celltype_col <- "Annotation_main_types_new"   # колонка с аннотацией 

args <- commandArgs(trailingOnly = TRUE)
objects <- if (length(args) > 0) args else c(
  # "data/seurat/ovary1.rds", "data/seurat/ovary2.rds", ...
)
if (length(objects) == 0) stop("передай .rds аргументами или впиши в objects")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
all_types <- character(0)

get_celltype <- function(obj) {
  md <- obj@meta.data
  if (celltype_col %in% colnames(md)) return(as.character(md[[celltype_col]]))
  message("  колонки '", celltype_col, "' нет, беру Idents()")
  as.character(Idents(obj))
}

get_coords <- function(obj) {
  xy <- GetTissueCoordinates(obj)
  xy <- as.data.frame(xy)
  cn <- tolower(colnames(xy))
  xi <- which(cn %in% c("x", "imagecol", "col", "x_centroid"))[1]
  yi <- which(cn %in% c("y", "imagerow", "row", "y_centroid"))[1]
  if (is.na(xi) || is.na(yi)) { xi <- 1; yi <- 2 }   
  data.frame(x = xy[[xi]], y = xy[[yi]])
}

for (path in objects) {
  message("читаю ", path)
  obj <- readRDS(path)
  slide <- tools::file_path_sans_ext(basename(path))

  ct <- get_celltype(obj)
  xy <- get_coords(obj)
  n <- min(nrow(xy), length(ct))

  df <- data.frame(
    cell_id = colnames(obj)[seq_len(n)],
    x = xy$x[seq_len(n)],
    y = xy$y[seq_len(n)],
    cell_type = ct[seq_len(n)]
  )
  out <- file.path(out_dir, paste0(slide, "_cells.csv"))
  write.csv(df, out, row.names = FALSE)
  message("  -> ", out, " (", nrow(df), " клеток)")
  all_types <- union(all_types, unique(df$cell_type))
}

# все типы - их надо разнести по группам в cell_type_map.yaml
message("\nвсе cell_type (", length(all_types), "):")
for (t in sort(all_types)) cat("  -", t, "\n")
writeLines(sort(all_types), file.path