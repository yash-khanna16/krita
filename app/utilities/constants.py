from enum import Enum


class SuccessMessages(Enum):
    HEALTH_CHECK_DONE = "Health check-up successful"

class ErrorMessages(Enum):
    SOMETHING_WENT_WRONG = "Oops! Something went wrong."
