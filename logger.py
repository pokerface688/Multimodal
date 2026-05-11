import logging
import logging.handlers
from pathlib import Path
import datetime

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    # 使用与文件日志相同的格式
    base_format = "[%(asctime)s] [%(levelname)s] [PID:%(process)d] [%(module)s:%(lineno)d] - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + base_format + reset,
        logging.INFO: grey + base_format + reset,
        logging.WARNING: yellow + base_format + reset,
        logging.ERROR: red + base_format + reset,
        logging.CRITICAL: bold_red + base_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

def setup_logging(
    log_dir: str = "logs",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    max_bytes: int = 10*1024*1024,  # 10MB
    backup_count: int = 200
):
    """
    配置全局日志系统
    
    :param log_dir: 日志目录路径
    :param console_level: 控制台日志级别
    :param file_level: 文件日志级别
    :param max_bytes: 单个日志文件最大大小
    :param backup_count: 保留的备份文件数量
    """
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 转化北京时间
    logging.Formatter.converter = beijing

    # 基础配置
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 清除已有handler
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # 定义日志格式
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [PID:%(process)d] [%(module)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 控制台Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level))
    console_handler.setFormatter(CustomFormatter())
    
    # 文件Handler（自动轮转）
    file_handler = logging.handlers.RotatingFileHandler(
        filename=Path(log_dir) / "1.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, file_level))
    file_handler.setFormatter(file_formatter)
    
    # 添加Handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


class ProjectLogger:
    def __init__(self, name: str = __name__):
        self.logger = logging.getLogger(name)
    
    def get_logger(self) -> logging.Logger:
        return self.logger

# 创建默认logger实例
default_logger = ProjectLogger(__name__).get_logger()