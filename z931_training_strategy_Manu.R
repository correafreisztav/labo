#Necesita para correr en Google Cloud
#  64 GB de memoria RAM
# 256 GB de espacio en el disco local
#   8 vCPU


#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")



#Parametros del script
PARAM  <- list()
PARAM$experimento <- "TS9310"

PARAM$exp_input  <- "FE9250"

PARAM$future       <- c( 202107 )

PARAM$final_train  <- c( 202101, 202102, 202103, 202104, 202105 )

PARAM$train$training     <- c( 202011, 202012, 202101, 202102, 202103 )
PARAM$train$validation   <- c( 202104 )
PARAM$train$testing      <- c( 202105 )
PARAM$train$undersampling  <- 1.0   # 1.0 significa NO undersampling ,  0.1  es quedarse con el 10% de los CONTINUA
PARAM$train$semilla  <- 131313