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
PARAM$experimento  <- "DD9990"
PARAM$exp_input  <- "HT9420"

PARAM$modelos  <- 1 #cuantos modelos quiero

# FIN Parametros del script

ksemilla  <- 131313

#------------------------------------------------------------------------------
options(error = function() { 
  traceback(20); 
  options(error = NULL); 
  stop("exiting after script error") 
})



particionar  <- function( data,  division, agrupa="",  campo="fold", start=1, seed=NA )
{
  if( !is.na(seed) )   set.seed( seed )
  
  bloque  <- unlist( mapply(  function(x,y) { rep( y, x )} ,   division,  seq( from=start, length.out=length(division) )  ) )  
  
  data[ , (campo) :=  sample( rep( bloque, ceiling(.N/length(bloque))) )[1:.N],
        by= agrupa ]
}


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Aqui empieza el programa

base_dir <- "~/buckets/b1/"

#creo la carpeta donde va el experimento
dir.create( paste0( base_dir, "exp/", PARAM$experimento, "/"), showWarnings = FALSE )
setwd(paste0( base_dir, "exp/", PARAM$experimento, "/"))   #Establezco el Working Directory DEL EXPERIMENTO

#leo la salida de la optimizaciob bayesiana
#arch_log  <- paste0( base_dir, "exp/", PARAM$exp_input, "/BO_log.txt" )
#tb_log  <- fread( arch_log )
#setorder( tb_log, -ganancia )

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


#######
parametrosDD <-  list( 
  boosting= "gbdt",               #puede ir  dart  , ni pruebe random_forest
  objective= "binary",
  metric= "custom",
  first_metric_only= TRUE,
  boost_from_average= TRUE,
  feature_pre_filter= FALSE,
  force_row_wise= TRUE,           #para que los alumnos no se atemoricen con tantos warning
  verbosity= -100,
  max_depth=  -1,                 # -1 significa no limitar,  por ahora lo dejo fijo
  min_gain_to_split= 0.0,         #por ahora, lo dejo fijo
  min_sum_hessian_in_leaf= 0.001, #por ahora, lo dejo fijo
  lambda_l1= 0.0,                 #por ahora, lo dejo fijo
  lambda_l2= 0.0,                 #por ahora, lo dejo fijo
  max_bin= 31,                    #por ahora, lo dejo fijo
  num_iterations= 5000,           #un numero muy grande, lo limita early_stopping_rounds
  
  bagging_fraction= 1.0,          #por ahora, lo dejo fijo
  pos_bagging_fraction= 1.0,      #por ahora, lo dejo fijo
  neg_bagging_fraction= 1.0,      #por ahora, lo dejo fijo
  
  drop_rate=  0.1,                #solo se activa en  dart
  max_drop= 50,                   #solo se activa en  dart
  skip_drop= 0.5,                 #solo se activa en  dart
  
  learning_rate = 0.2,
  early_stopping_rounds = 4000,   # Corta cuando despuess de tantos arboles no vio una ganancia mejor a la maxima
  feature_fraction = .7,
  
  extra_trees= FALSE,
  
  seed=  ksemilla
)
######

#genero un modelo para cada uno de las modelos_qty MEJORES iteraciones de la Bayesian Optimization
for( i in  1:PARAM$modelos )
{
  parametros  <- parametrosDD
  #iteracion_bayesiana  <- parametros$iteracion_bayesiana
  
  arch_modelo  <- paste0( "modelo_" ,
                          sprintf( "%02d", i ),
                          ".model" )
  
  
  #creo CADA VEZ el dataset de lightgbm
  dtrain  <- lgb.Dataset( data=    data.matrix( dataset[ , campos_buenos, with=FALSE] ),
                          label=   dataset[ , clase01],
                          weight=  dataset[ , ifelse( clase_ternaria %in% c("BAJA+2"), 1.0000001, 1.0)],
                          free_raw_data= FALSE
  )
  
  particionar( dtrain, division=c(7,3), agrupa="clase_ternaria", seed= ksemilla )
  
  #Utilizo la semilla definida en este script
  parametros$seed  <- ksemilla
  
  #genero el modelo entrenando en los datos finales
  set.seed( parametros$seed )
  # modelo  <- lightgbm( data= dtrain[[fold==1]], #uso fold 1 como train
  #                            param=  parametros,
  #                            verbose= -100 )
  modelo  <- lightgbm( data= dtrain[fold==1],
                       valids= dtrain[fold==2],
                       param=  parametros,
                       verbose= -100 )
  
  #grabo el modelo, achivo .model
  lgb.save( modelo,
            file= arch_modelo )
  
  #creo y grabo la importancia de variables
  tb_importancia  <- as.data.table( lgb.importance( modelo ) )
  fwrite( tb_importancia,
          file= paste0( "impo_", 
                        sprintf( "%02d", i ),
                        ".txt" ),
          sep= "\t" )
  
  #genero la prediccion, Scoring
  prediccion  <- predict( modelo,
                          data.matrix( dfuture[ , campos_buenos, with=FALSE ] ) )
  
  #aplico el modelo a los datos de testing
  pred_val =predict( modelo,   #el modelo que genere recien
                     dataset[ fold==2],  #fold==2  es validacion. 
  )
  
  
  tb_prediccion  <- dfuture[  , list( numero_de_cliente, foto_mes ) ]
  tb_prediccion[ , prob := prediccion ]
  
  
  nom_pred  <- paste0( "pred_",
                       sprintf( "%02d", i ),
                       ".csv"  )
  
  fwrite( tb_prediccion,
          file= nom_pred,
          sep= "\t" )
  
  
  #genero los archivos para Kaggle
  cortes  <- seq( from=  7000,
                  to=   11000,
                  by=     500 )
  
  
  setorder( tb_prediccion, -prob )
  
  for( corte in cortes )
  {
    tb_prediccion[  , Predicted := 0L ]
    tb_prediccion[ 1:corte, Predicted := 1L ]
    
    nom_submit  <- paste0( PARAM$experimento, 
                           "_",
                           sprintf( "%02d", i ),
                           "_",
                           sprintf( "%05d", corte ),
                           ".csv" )
    
    fwrite(  tb_prediccion[ , list( numero_de_cliente, Predicted ) ],
             file= nom_submit,
             sep= "," )
    
  }
  
  
  #borro y limpio la memoria para la vuelta siguiente del for
  rm( tb_prediccion )
  rm( tb_importancia )
  rm( modelo)
  rm( parametros )
  rm( dtrain )
  gc()
}
