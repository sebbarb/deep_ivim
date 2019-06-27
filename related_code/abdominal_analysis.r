library(psych)

#df = read.csv("H:/projects/IVIM/code/deep_learning_IVIM_autoencoder/data/df_for_icc.csv", header = TRUE)
df = read.csv("H:/projects/IVIM/code/deep_learning_IVIM_autoencoder/data/df_for_icc_3T.csv", header = TRUE)

print(ICC(df[, c('Dp_LS_Andres', 'Dp_LS_Olivio')]))
print(ICC(df[, c('Dp_BP_Andres', 'Dp_BP_Olivio')]))
print(ICC(df[, c('Dp_NN_Andres', 'Dp_NN_Olivio')]))

print(ICC(df[, c('Dt_LS_Andres', 'Dt_LS_Olivio')]))
print(ICC(df[, c('Dt_BP_Andres', 'Dt_BP_Olivio')]))
print(ICC(df[, c('Dt_NN_Andres', 'Dt_NN_Olivio')]))

print(ICC(df[, c('Fp_LS_Andres', 'Fp_LS_Olivio')]))
print(ICC(df[, c('Fp_BP_Andres', 'Fp_BP_Olivio')]))
print(ICC(df[, c('Fp_NN_Andres', 'Fp_NN_Olivio')]))

