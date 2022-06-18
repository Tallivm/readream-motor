from dataset import DatasetBuilder
from models import NNModelTrainer
from plotting import plot_loss_curves, plot_confusion_matrix

if __name__ == "__main__":

    databuilder = DatasetBuilder()
    databuilder.collect_datasets()
    databuilder.prepare_spatial_remapping()
    databuilder.prepare_batches()

    trainer = NNModelTrainer(databuilder)
    trainer.get_model()
    trainer.train_and_test()
    plot_loss_curves(databuilder.runtime, trainer.loss['train'], trainer.loss['validate'])

    y_true, y_pred = trainer.test()
    plot_confusion_matrix(databuilder.runtime, y_true, y_pred)
