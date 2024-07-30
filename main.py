from flask import Flask
from flask_restful import Api
from old_extraction import ImageExtractor


app = Flask(__name__)
api = Api(app)

api.add_resource(ImageExtractor, '/<file_name>')