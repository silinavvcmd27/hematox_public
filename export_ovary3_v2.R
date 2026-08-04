# Извлечение клеток ovary3 (xenium4) из аннотированного Seurat-объекта.
# Комбинированный тип: подтип CAF для фибробластов, иначе основной тип —
# так в таблицу попадают и C7+ CAFs (гормональная), и mCAFs (матриксная).
#
#   Rscript export_ovary3_v2.R

suppressMessages({library(Seurat); library(dplyr)})

path <- "/mnt/singlecellproject/public_data/Ovarian_cancer/xenium4/R_obj/seurat_ann_2.rds"
obj  <- UpdateSeuratObject(readRDS(path))

cat("=== колонки аннотации ===\n")
print(grep("Annotation|ann", colnames(obj@meta.data), value = TRUE, ignore.case = TRUE))

# комбинированный тип
main <- as.character(obj$Annotation_main_types_new)
sub  <- as.character(obj$Annotation_2_iter)
fib  <- as.character(obj$Annotation_common) == "Fibroblasts"
ct <- main
ct[fib] <- sub[fib]
ct[is.na(ct)] <- "NA"

cat("\n=== типы клеток (комбинированные) ===\n")
print(sort(table(ct), decreasing = TRUE))

# координаты клеток (центроиды Xenium)
co <- GetTissueCoordinates(obj)
cat("\n=== координаты ===\n")
cat("колонки:", paste(colnames(co), collapse = ", "), "\n")
print(head(co))
xcol <- if ("x" %in% colnames(co)) "x" else colnames(co)[1]
ycol <- if ("y" %in% colnames(co)) "y" else colnames(co)[2]
cat(sprintf("x: %.0f .. %.0f | y: %.0f .. %.0f | строк: %d\n",
            min(co[[xcol]]), max(co[[xcol]]),
            min(co[[ycol]]), max(co[[ycol]]), nrow(co)))

# сборка таблицы с сопоставлением по cell_id
ids  <- colnames(obj)
cell <- if ("cell" %in% colnames(co)) as.character(co$cell) else rownames(co)
out <- data.frame(cell_id = cell,
                  x = co[[xcol]], y = co[[ycol]],
                  stringsAsFactors = FALSE)
out$cell_type <- ct[match(out$cell_id, ids)]
out <- out[!is.na(out$cell_type) & out$cell_type != "NA", ]

dir.create("data/seurat_csv", showWarnings = FALSE, recursive = TRUE)
outpath <- "data/seurat_csv/ovary3_v2_cells.csv"
write.csv(out, outpath, row.names = FALSE)
cat("\nсохранено:", outpath, "| клеток:", nrow(out), "\n")