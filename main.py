from app import CreateApp
from config import PORT, HOST
from app.utilities.logger import logger
from app.utilities.responseHandler import failure_response


# Creating the app
app = CreateApp().create_app()


if __name__ == '__main__':

    logger.info(f"App started successfully on - {HOST}:{PORT}")

    # Running the app
    app.run(host=HOST, port=PORT)


    @app.errorhandler(404)
    def not_found(error):
        return failure_response('Not found', 404)
