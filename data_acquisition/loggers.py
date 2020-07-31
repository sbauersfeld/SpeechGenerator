import sys
from io import TextIOWrapper
from typing import Optional, TextIO


class Logger(TextIOWrapper):
  def __init__(self, log_file, stream) -> None:
    self.__stream = stream
    self.__log_file = log_file

  def write(self, x) -> int:
    self.__log_file.write(x)
    self.__stream.write(x)
    self.__log_file.flush()
    self.__stream.flush()

    return len(x)


class LoggerFactory():
  def __init__(self, log_file_loc) -> None:
    self.__log_file: Optional[TextIO]
    if log_file_loc:
      self.__log_file = open(log_file_loc, "w", encoding="utf-8")
    else:
      self.__log_file = None

  def __enter__(self) -> "LoggerFactory":
    return self
  
  def __exit__(self, exc_type, exc_value, traceback) -> None:
    if self.__log_file:
      self.__log_file.close()
  
  def get_logger(self, stream) -> Logger:
    if self.__log_file:
      return Logger(self.__log_file, stream)
    else:
      return stream

  def set_loggers(self) -> None:
    sys.stdout = self.get_logger(sys.stdout)
    sys.stderr = self.get_logger(sys.stderr)
