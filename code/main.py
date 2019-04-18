###################### load packages ####################
import config
import dataset
import train


###################### main函数 ####################
def main():

    ########### 读取配置文件 ##########
    ch = config.ConfigHandler("./config.ini")
    ch.load_config()

    ########### 读取参数 ##########
    batch_size = int(ch.config["model"]["batch_size"])
    num_epochs = int(ch.config["model"]["num_epochs"])
    learning_rate = float(ch.config["model"]["learning_rate"])
    class_size = int(ch.config["model"]["class_size"])
    log_interval = int(ch.config["log"]["log_interval"])

    ########### 获取数据loader ##########
    data_loader_train = dataset.MyDataset(batch_size).load_train_data()
    data_loader_test = dataset.MyDataset(batch_size).load_test_data()

    ########### 训练和评价 ##########
    train.train_and_test().train_epoch(num_epochs, learning_rate, class_size, data_loader_train, data_loader_test, log_interval)


if __name__ == "__main__":
    main()
