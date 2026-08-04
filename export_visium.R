# координаты спотов + метка tumor/stroma + H&E
# запуск: Rscript export_visium.R <rds> <spatial_dir> <out_name>
suppressMessages(library(Seurat))
a   <- commandArgs(trailingOnly = TRUE)
rds <- a[1]; sdir <- a[2]; name <- a[3]
out <- "data/processed/visium"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

b <- UpdateSeuratObject(readRDS(rds))

# метка из деконволюции: доминирующий компартмент
cols <- c("Epithelial.cells", "Stromal.cells", "Immune.cells",
          "Endothelial.cells", "Perycites")
cols <- cols[cols %in% colnames(b@meta.data)]
cat("деконволюция по:", paste(cols, collapse = ", "), "\n")
S   <- as.matrix(b@meta.data[, cols])
top <- cols[max.col(S, ties.method = "first")]
lab <- setNames(ifelse(top == "Epithelial.cells", "tumor", "stroma"),
                rownames(b@meta.data))
cat("tumor:", sum(lab == "tumor"), " stroma:", sum(lab == "stroma"), "\n")

# координаты спотов (полноразмерное пространство слайда)
co <- GetTissueCoordinates(b)
sf <- b@images[[1]]@scale.factors
df <- data.frame(barcode = co$cell, x_full = co$x, y_full = co$y,
                 label = lab[co$cell], epi = S[co$cell, "Epithelial.cells"],
                 hires_scale = sf$hires, spot_full_px = sf$spot,
                 stringsAsFactors = FALSE)
write.csv(df, file.path(out, paste0(name, "_spots.csv")), row.names = FALSE)

# H&E: hires png — лучшее, что есть в SpaceRanger
file.copy(file.path(sdir, "spatial", "tissue_hires_image.png"),
          file.path(out, paste0(name, "_he.png")), overwrite = TRUE)
cat("готово:", name, "->", out, "\n")