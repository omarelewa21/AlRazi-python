from flask import Flask
from flask_restful import Api
from image_extractor import ImageExtractor


app = Flask(__name__)
api = Api(app)

api.add_resource(ImageExtractor, '/<file_name>')