#Aplicacion de los mejores hiperparametros encontrados en una bayesiana
#Utilizando clase_binaria =  [  SI = { "BAJA+1", "BAJA+2"} ,  NO="CONTINUA ]

#cargo las librerias que necesito
require("data.table")
require("rpart")
require("rpart.plot")
library(dplyr)


#Aqui se debe poner la carpeta de la materia de SU computadora local
setwd("C:/Users/manu-/Documents/R maestria/labo")  #Establezco el Working Directory

#cargo el dataset
dataset  <- fread("./datasets/competencia1_2022.csv" )


#creo la clase_binaria SI={ BAJA+1, BAJA+2 }    NO={ CONTINUA }
dataset[ foto_mes==202101, 
         clase_binaria :=  ifelse( clase_ternaria=="CONTINUA", "NO", "SI" ) ]


#agrego una variable canarito, random distribucion uniforme en el intervalo [0,1]
dataset[ ,  canarito1 :=  runif( nrow(dataset) ) ]

dataset[, 
        feature :=  Master_mfinanciacion_limite/Visa_msaldototal ]
#sumar todo plata a favor y en contra
#distancia de la media
#avisar en zulip que no me cambió nada la diferencia entre meses
#deudas o algo asi
dataset[, 
        feature2 :=  mcaja_ahorro*ctrx_quarter ]

dataset[, 
        prestamodeuda :=  mprestamos_personales+cprestamos_personales/mcuentas_saldo ]

dataset<-transform(dataset,negativo=ifelse(0>mcuentas_saldo,1,0))

dataset[, 
        numeros :=  rowSums(select_if(dataset, is.numeric),na.rm = TRUE) ]
dataset[, 
        viejoyconfiable :=  mrentabilidad_annual/cliente_edad ]


dtrain  <- dataset[ foto_mes==202101 ]  #defino donde voy a entrenar
dapply  <- dataset[ foto_mes==202103 ]  #defino donde voy a aplicar el modelo


# Entreno el modelo
# obviamente rpart no puede ve  clase_ternaria para predecir  clase_binaria
#  #no utilizo Visa_mpagado ni  mcomisiones_mantenimiento por drifting



#----------------------------------------------------------------------------
# habilitar esta seccion si el Fiscal General  Alejandro Bolaños  lo autoriza
#----------------------------------------------------------------------------

#corrijo manualmente el drifting de  Visa_fultimo_cierre
dapply[ Visa_fultimo_cierre== 1, Visa_fultimo_cierre :=  4 ]
dapply[ Visa_fultimo_cierre== 7, Visa_fultimo_cierre := 11 ]
dapply[ Visa_fultimo_cierre==21, Visa_fultimo_cierre := 25 ]
dapply[ Visa_fultimo_cierre==14, Visa_fultimo_cierre := 18 ]
dapply[ Visa_fultimo_cierre==28, Visa_fultimo_cierre := 32 ]
dapply[ Visa_fultimo_cierre==35, Visa_fultimo_cierre := 39 ]
dapply[ Visa_fultimo_cierre> 39, Visa_fultimo_cierre := Visa_fultimo_cierre + 4 ]

#corrijo manualmente el drifting de  Visa_fultimo_cierre
dapply[ Master_fultimo_cierre== 1, Master_fultimo_cierre :=  4 ]
dapply[ Master_fultimo_cierre== 7, Master_fultimo_cierre := 11 ]
dapply[ Master_fultimo_cierre==21, Master_fultimo_cierre := 25 ]
dapply[ Master_fultimo_cierre==14, Master_fultimo_cierre := 18 ]
dapply[ Master_fultimo_cierre==28, Master_fultimo_cierre := 32 ]
dapply[ Master_fultimo_cierre==35, Master_fultimo_cierre := 39 ]
dapply[ Master_fultimo_cierre> 39, Master_fultimo_cierre := Master_fultimo_cierre + 4 ]



#agrego 45 canaritos
for( i in 1:45 ) dataset[ , paste0("canarito", i ) :=  runif( nrow(dataset)) ]

modelo  <- rpart(formula=   "clase_binaria ~ . -clase_ternaria -Master_fultimo_cierre -mcomisiones_mantenimiento",
                 data=      dtrain,  #los datos donde voy a entrenar
                 xval=         0,
                 cp=          -0.81,#  -0.89
                 minsplit=  640,   # 621
                 minbucket=  309,   # 309
                 maxdepth=     9 )  #  10


modelo$frame[ modelo$frame$var %like% "canarito", "complexity"] <- -666
modelo_pruned  <- prune(  modelo, -666 )


prediccion  <- predict( modelo_pruned, dapply, type = "prob")[,"NO"]

entrega  <-  as.data.table( list( "numero_de_cliente"= dapply$numero_de_cliente,
                                  "Predicted"= as.integer(  prediccion > 0.025 ) ) )

fwrite( entrega, paste0( "./exp/pajaritos/stopping_at_canaritos.csv"), sep="," )

pdf(file = "./exp/pajaritos/stopping_at_canaritos.pdf", width=28, height=4)
prp(modelo_pruned, extra=101, digits=5, branch=1, type=4, varlen=0, faclen=0)
dev.off()


#prediccion es una matriz con DOS columnas, llamadas "NO", "SI"
#cada columna es el vector de probabilidades 

#agrego a dapply una columna nueva que es la probabilidad de BAJA+2
dfinal  <- copy( dapply[ , list(numero_de_cliente) ] )
dfinal[ , prob_SI := prediccion[ , "SI"] ]


# por favor cambiar por una semilla propia
semillas = list(3,13,31,113,331)
# que sino el Fiscal General va a impugnar la prediccion
set.seed(113)
dfinal[ , azar := runif( nrow(dapply) ) ]

# ordeno en forma descentente, y cuando coincide la probabilidad, al azar
setorder( dfinal, -prob_SI, azar )


dir.create( "./exp/" )
dir.create( "./exp/KA4120" )


#grafico el arbol
prp(modelo, extra=101, digits=5, branch=1, type=4, varlen=0, faclen=0)

for( corte  in  c( 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000 ) )
{
  #le envio a los  corte  mejores,  de mayor probabilidad de prob_SI
  dfinal[ , Predicted := 0L ]
  dfinal[ 1:corte , Predicted := 1L ]
  
  
  fwrite( dfinal[ , list(numero_de_cliente, Predicted) ], #solo los campos para Kaggle
          file= paste0( "./exp/KA4120/KA4120_005_",  corte, ".csv"),
          sep=  "," )
}

summary(modelo)
names(modelo$variable.importance)

