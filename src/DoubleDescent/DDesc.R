#Necesita para correr en Google Cloud
# 128 GB de memoria RAM
# 256 GB de espacio en el disco local
#   8 vCPU

# ZZ final que necesita de UNDERSAMPLING

#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")

require("lightgbm")

#Parametros del script
PARAM  <- list()
PARAM$experimento  <- "DD9999"
PARAM$exp_input  <- "HT9420"

PARAM$modelos  <- 3 #cuantos modelos quiero
PARAM$piso    <-    50  #solucion rapida para adicionar desde donde arranco
secuencia  <- seq( from=  100, #secuencia solo se usará si es necesario
                   to=   1000,
                   by=     50 )



PARAM$finalmodel$max_bin           <-     31
PARAM$finalmodel$learning_rate     <-      0.0280015981   #0.0142501265
PARAM$finalmodel$num_iterations    <-    50  #porque voy de a steps de 100 en complejidad
PARAM$finalmodel$num_leaves        <-   1015  #784
PARAM$finalmodel$min_data_in_leaf  <-   5542  #5628
PARAM$finalmodel$feature_fraction  <-      0.7832319551  #0.8382482539
PARAM$finalmodel$semilla           <- 131313

# FIN Parametros del script

ksemilla  <- 131313

#------------------------------------------------------------------------------
options(error = function() { 
  traceback(20); 
  options(error = NULL); 
  stop("exiting after script error") 
})
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Aqui empieza el programa

base_dir <- "~/buckets/b1/"

#creo la carpeta donde va el experimento
dir.create( paste0( base_dir, "exp/", PARAM$experimento, "/"), showWarnings = FALSE )
setwd(paste0( base_dir, "exp/", PARAM$experimento, "/"))   #Establezco el Working Directory DEL EXPERIMENTO

#leo la salida de la optimizaciob bayesiana
arch_log  <- paste0( base_dir, "exp/", PARAM$exp_input, "/BO_log.txt" )
tb_log  <- fread( arch_log )
setorder( tb_log, -ganancia )

#leo el nombre del expermento de la Training Strategy
arch_TS  <- paste0( base_dir, "exp/", PARAM$exp_input, "/TrainingStrategy.txt" )
TS  <- readLines( arch_TS, warn=FALSE )

#leo el dataset donde voy a entrenar el modelo final
arch_dataset  <- paste0( base_dir, "exp/", TS, "/dataset_train_final.csv.gz" )
dataset  <- fread( arch_dataset )

#leo el dataset donde voy a aplicar el modelo final
arch_future  <- paste0( base_dir, "exp/", TS, "/dataset_future.csv.gz" )
dfuture <- fread( arch_future )

#defino la clase binaria
dataset[ , clase01 := ifelse( clase_ternaria %in% c("BAJA+1","BAJA+2"), 1, 0 )  ]

campos_buenos  <- setdiff( colnames(dataset), c( "clase_ternaria", "clase01") )

#genero un modelo para cada uno de las modelos_qty MEJORES iteraciones de la Bayesian Optimization
for( i in  1:PARAM$modelos )
{
  parametros  <- as.list( copy( tb_log[ i ] ) )
  iteracion_bayesiana  <- parametros$iteracion_bayesiana
  
  arch_modelo  <- paste0( "modelo_steps" ,
                          sprintf( "%02d", PARAM$piso+i*PARAM$finalmodel$num_iterations ),
                          ".model" )
  
  #creo CADA VEZ el dataset de lightgbm
  dtrain  <- lgb.Dataset( data=    data.matrix( dataset[ , campos_buenos, with=FALSE] ),
                          label=   dataset[ , clase01],
                          weight=  dataset[ , ifelse( clase_ternaria %in% c("BAJA+2"), 1.0000001, 1.0)],
                          free_raw_data= FALSE
  )
  
  ganancia  <- parametros$ganancia
  
  #elimino los parametros que no son de lightgbm
  parametros$experimento  <- NULL
  parametros$cols         <- NULL
  parametros$rows         <- NULL
  parametros$fecha        <- NULL
  parametros$prob_corte   <- NULL
  parametros$estimulos    <- NULL
  parametros$ganancia     <- NULL
  parametros$iteracion_bayesiana  <- NULL
  
  if( ! ("leaf_size_log" %in% names(parametros) ) )  stop( "El Hyperparameter Tuning debe tener en BO_log.txt  el pseudo hiperparametro  lead_size_log.\n" )
  if( ! ("coverage" %in% names(parametros) ) ) stop( "El Hyperparameter Tuning debe tener en BO_log.txt  el pseudo hiperparametro  coverage.\n" )
  
  #Primero defino el tamaño de las hojas
  #parametros$min_data_in_leaf  <- pmax( 1,  round( nrow(dtrain) / ( 2.0 ^ parametros$leaf_size_log ))  )
  #Luego la cantidad de hojas en funcion del valor anterior, el coverage, y la cantidad de registros
  #parametros$num_leaves  <-  pmin( 131072, pmax( 2,  round( parametros$coverage * nrow( dtrain ) / parametros$min_data_in_leaf ) ) )
  #cat( "min_data_in_leaf:", parametros$min_data_in_leaf,  ",  num_leaves:", parametros$num_leaves, "\n" )
  
  #ya no me hacen falta
  parametros$leaf_size_log  <- NULL
  parametros$coverage  <- NULL
  
  #Utilizo la semilla definida en este script
  parametros$seed  <- ksemilla
  
  param= list( objective=          "binary",
               max_bin=            PARAM$finalmodel$max_bin,
               learning_rate=      PARAM$finalmodel$learning_rate,
               num_iterations=     PARAM$piso+i*(PARAM$finalmodel$num_iterations),
               num_leaves=         PARAM$finalmodel$num_leaves,
               min_data_in_leaf=   PARAM$finalmodel$min_data_in_leaf,
               feature_fraction=   PARAM$finalmodel$feature_fraction,
               seed=               PARAM$finalmodel$semilla
  )
  
  #genero el modelo entrenando en los datos finales
  set.seed( parametros$seed )
  modelo_final  <- lightgbm( data= dtrain,
                             param=  param,
                             verbose= -100 )
  
  #grabo el modelo, achivo .model
  lgb.save( modelo_final,
            file= arch_modelo )
  
  #creo y grabo la importancia de variables
  tb_importancia  <- as.data.table( lgb.importance( modelo_final ) )
  fwrite( tb_importancia,
          file= paste0( "impo_", 
                        sprintf( "%02d", PARAM$piso+i*PARAM$finalmodel$num_iterations ),
                        ".txt" ),
          sep= "\t" )
  
  
  #genero la prediccion, Scoring
  prediccion  <- predict( modelo_final,
                          data.matrix( dfuture[ , campos_buenos, with=FALSE ] ) )
  
  tb_prediccion  <- dfuture[  , list( numero_de_cliente, foto_mes ) ]
  tb_prediccion[ , prob := prediccion ]
  
  
  nom_pred  <- paste0( "pred_",
                       sprintf( "%02d", PARAM$piso+i*PARAM$finalmodel$num_iterations ),
                       ".csv"  )
  
  fwrite( tb_prediccion,
          file= nom_pred,
          sep= "\t" )
  
  
  #genero los archivos para Kaggle
  cortes  <- seq( from=  9000,
                  to=   11000,
                  by=     1000 )
  
  
  setorder( tb_prediccion, -prob )
  
  for( corte in cortes )
  {
    tb_prediccion[  , Predicted := 0L ]
    tb_prediccion[ 1:corte, Predicted := 1L ]
    
    nom_submit  <- paste0( PARAM$experimento, 
                           "_",
                           sprintf( "%02d", i ),
                           "_",
                           sprintf( "%03d", PARAM$piso+i*PARAM$finalmodel$num_iterations),
                           "_",
                           sprintf( "%05d", corte ),
                           ".csv" )
    
    fwrite(  tb_prediccion[ , list( numero_de_cliente, Predicted ) ],
             file= nom_submit,
             sep= "," )
    
  }
  
  print(PARAM$piso+i*PARAM$finalmodel$num_iterations)
  #borro y limpio la memoria para la vuelta siguiente del for
  rm( tb_prediccion )
  rm( tb_importancia )
  rm( modelo_final)
  rm( parametros )
  rm( dtrain )
  gc()
}