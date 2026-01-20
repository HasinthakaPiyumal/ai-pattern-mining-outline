# Cluster 0

def main():
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()
    model = build_model(data_handler).to(configs['device'])
    logger = Logger()
    trainer = build_trainer(data_handler, logger)
    trainer.train(model)

