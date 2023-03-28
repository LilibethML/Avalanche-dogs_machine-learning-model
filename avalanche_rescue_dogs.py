# En este modelo, lo que queremos es que nos estime la talla de las botas 
# de perros de rescate en funcion de la talla de su arnés. 
# harness_size va a ser nuestro input , y queremos un modelo que procese el
# input y de sus estimaciones de la talla de las botas (output)
import pandas


# Making a dictionary of data for boot sizes
# and harness size in cm
data = {
    'boot_size' : [ 39, 38, 37, 39, 38, 35, 37, 36, 35, 40, 
                    40, 36, 38, 39, 42, 42, 36, 36, 35, 41, 
                    42, 38, 37, 35, 40, 36, 35, 39, 41, 37, 
                    35, 41, 39, 41, 42, 42, 36, 37, 37, 39,
                    42, 35, 36, 41, 41, 41, 39, 39, 35, 39
 ],
    'harness_size': [ 58, 58, 52, 58, 57, 52, 55, 53, 49, 54,
                59, 56, 53, 58, 57, 58, 56, 51, 50, 59,
                59, 59, 55, 50, 55, 52, 53, 54, 61, 56,
                55, 60, 57, 56, 61, 58, 53, 57, 57, 55,
                60, 51, 52, 56, 55, 57, 58, 57, 51, 59
                ]
}

# Converting it into a table using pandas
dataset = pandas.DataFrame(data)

# Printing the data
print(dataset)
#----------------------------------------------------------------------------
# Hecho el dataframe, seleccionamos un modelo. Escogemos el OLS, que es una 
# regresion lineal de minimos cuadrados. Esto porque queremos encontrar una relacion
# lineal entre los arneses y las botas. 

# Loading a library to do the hard work for us
import statsmodels.formula.api as smf

# First, we define our formula using a special syntax
# This says that boot_size is explained by harness_size
formula = "boot_size ~ harness_size"

# Creating the model, but don't train it yet
model = smf.ols(formula = formula, data = dataset)

# We can see that we have created our model but it does not 
# have internal parameters set yet
if not hasattr(model, 'params'):
    print("Model selected but it does not have parameters set. We need to train it!")

 # Entrenando el modelo--------------------------------------------------------
 # Los modelos OLS necesitan dos parametros (m y b). Necesitamos entrenar al modelo
 # (fit) para encontrar esos parametros. Observe cómo el entrenamiento del modelo 
 # establece sus parámetros. 

 # Loading some libraries to do the hard work for us
import graphing 

# Training (fitting) the model so that it creates a line that 
# fits our data. This method does the hard work for us
fitted_model = model.fit()

# Printing information about our model now it has been fit
print("The following model parameters have been found:\n" +
        f"Line slope: {fitted_model.params[1]}\n"+
        f"Line Intercept: {fitted_model.params[0]}")

#Ahora grafiquemos eso --------------------------------------------------------
# Showing a graph of the result. 
graphing.scatter_2D(dataset,    label_x="harness_size", 
                                label_y="boot_size",
                                trendline=lambda x: fitted_model.params[1] * x + fitted_model.params[0]
                                )

 # En la grafica se observan los circulos, los cuales son los datos originales. 
 # La linea roja es nuestro modelo. 

 #Uso del modelo----------------------------------------------------------------
 # Ahora que terminó el entrenamiento, podemos hacer uso de nuestro modelo para
 # predecir cualquier talla de botas a partir de talla de arnés

 # harness_size states the size of the harness we are interested in
harness_size = { 'harness_size' : [52.5] }

# Using the model to predict what size of boots the dog will fit
approximate_boot_size = fitted_model.predict(harness_size)

# Printing the result
print("Estimated approximate_boot_size:")
print(approximate_boot_size[0])                               