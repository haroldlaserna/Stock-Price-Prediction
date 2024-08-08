from evaluate import *
from modelization import *
from util import *
from preparation import *

def main():
    # buscar link
    config_path = os.path.join('config', 'initial_parameters.json')
    config_model = os.path.join('config', 'models.json')
    
    # Cargar configuración
    config = load_config(config_path)
    models = load_models_from_config(config_model)
    
    # Acceder a los valores del JSON
    symbol = config.get('symbol')
    start_date = config.get('start_date')
    end_date = config.get('end_date')
    seq_length = config.get('seq_length')
    month = config.get('month')
    secondary_symbols = config.get('secondary_symbols')

    #Limpiar terminal
    clear_terminal()

    # Obtener y preparar datos
    data, data1 = get_and_prepare_data(symbol, secondary_symbols, start_date, end_date, month)
    
    # Preparar los datos para el modelo
    X_train, X_test, y_train, y_test, scaler_x, scaler_y = prepare_model_data(data, seq_length)
    
    predictions = []
    model_names = []
    # Crear y entrenar los modelos y evaluar modelos
    for model_name, create_model_func in models.items():
        print(f"Entrenando el modelo {model_name}...")
        model, history = create_and_train_model(create_model_func, X_train, y_train, X_test, y_test)
        Y_prediccion, history = transform_and_evaluate_predictions(model, data1, scaler_x, scaler_y, seq_length, history)
        predictions.append(Y_prediccion)
        model_names.append(model_name)
        plot_loss(history)
        
    # Graficar todas las predicciones
    plot_all_predictions(data1, predictions, model_names, symbol, seq_length)
    
    
if __name__ == '__main__':
    main()
    