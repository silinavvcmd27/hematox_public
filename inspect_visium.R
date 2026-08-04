# понять структуру Seurat-объектов Visium
suppressMessages(library(Seurat))

path <- commandArgs(trailingOnly = TRUE)[1]
b <- UpdateSeuratObject(readRDS(path))

cat("===== meta.data колонки =====\n")
print(colnames(b@meta.data))

cat("\n===== кандидаты на аннотацию (мало уникальных значений) =====\n")
for (col in colnames(b@meta.data)) {
  v <- b@meta.data[[col]]
  if (is.factor(v) || is.character(v)) {
    u <- unique(v)
    if (length(u) <= 15)
      cat(sprintf("  %s  ->  %s\n", col, paste(u, collapse = ", ")))
  }
}

cat("\n===== images =====\n")
print(names(b@images))
im <- b@images[[1]]
cat("класс:", class(im)[1], "\n")
cat("размер встроенной картинки:", paste(dim(im@image), collapse = " x "), "\n")
cat("scale.factors:\n"); print(im@scale.factors)

cat("\n===== координаты спотов =====\n")
co <- GetTissueCoordinates(b)
print(head(co))
cat("row:", paste(range(co[, 1]), collapse = " .. "),
    "| col:", paste(range(co[, 2]), collapse = " .. "),
    "| спотов:", nrow(co), "\n")