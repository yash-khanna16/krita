"""This module is the core of the project."""
from flask import Flask
from flask_cors import CORS

from app.utilities.logger import init_logger
from config import TOKEN_SECRET
from app.jd_vector_search.services import JdVectorSearch

# Creating an object of Vector Search class

obj_jd_search = JdVectorSearch()


class CreateApp:
    """
    Initialize the core application
    """
    app = Flask(__name__, template_folder='templates')

    def __init__(self):
        self.headers = None

    def create_app(self):
        self.app.secret_key = TOKEN_SECRET
        self.app.config.from_pyfile('../config.py')

        # Setting CORS
        CORS(self.app)

        # Initiating logger
        init_logger(self.app)

        with self.app.app_context():
            # Import a module/component using its blueprint handler variable
            from app.openai.controllers import mod_openai as openai_module
            from app.jd_vector_search.controllers import mod_vector_search
            from app.resume_parser.controllers import mod_resume_parser
            from app.content_analyzer.controllers import mod_content_analyzer
            from app.persona.controllers import mod_persona

            # Register blueprint(s)
            self.app.register_blueprint(openai_module)
            self.app.register_blueprint(mod_vector_search)
            self.app.register_blueprint(mod_resume_parser)
            self.app.register_blueprint(mod_content_analyzer)
            self.app.register_blueprint(mod_persona)

            return self.app

    @app.after_request
    def set_headers(self):
        # Setting Header Options
        self.headers['X-Frame-Options'] = 'SAMEORIGIN'
        self.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        self.headers["Pragma"] = "no-cache"  # HTTP 1.0.
        self.headers["Expires"] = "0"  # Proxies
        self.headers['X-Content-Type-Options'] = 'nosniff'

        return self
