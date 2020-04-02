import logging
import time


def logger_init():
    # Step1 : 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Step2 : 创建一个handler，用于写入日志文件
    time_stamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logfile = './logs/' + time_stamp + '.log'

    # 用于输出到文件的handler
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.INFO)  # 输出到file的log等级的开关
    # 用于输出到控制台的handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # Step3 : 定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s : %(message)s")
    # formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    console.setFormatter(formatter)

    # Step4 : 将logger添加到handler里面
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger
