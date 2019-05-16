import math
import argparse
from flask_restplus import Api, Resource, fields
from flask import Flask, jsonify, request, make_response, abort, render_template, redirect, url_for
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from flask_restplus import Api, Resource, fields
from flask import Flask, request, jsonify
import numpy as np
from werkzeug.datastructures import FileStorage
from PIL import Image
from keras.models import model_from_json
import tensorflow as tf
from keras.models import load_model
import time

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate('ups-warehouse-app-firebase-adminsdk-97yy8-2a2c26ad1b.json')
firebase_admin.initialize_app(cred)

db = firestore.client()
print('About to print users')
users_ref = db.collection(u'ds23_flask')
docs = users_ref.get()
# print("Number of items", str(len(docs)))
for doc in docs:
    print(u'{} => {}'.format(doc.id, doc.to_dict()))


application = Flask(__name__)
api = Api(application, version='1.0', title='MNIST Classification', description='CNN for Mnist')
ns = api.namespace('Make_School', description='Methods')

single_parser = api.parser()
single_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)

model = load_model('my_model.h5')
graph = tf.get_default_graph()
#
# with open('model_architecture.json', 'r') as f:
#     new_model_1 = model_from_json(f.read())
# new_model_1.load_weights('model_weights.h5')


# def cylinder_volume(radius, height):
#     vol = (math.pi) * (radius ** 2)  * height
#     return vol
#
#
# def summation(a, b):
#     return a+b

# @ns.route('/addition')
# class Addition(Resource):
#     @api.doc(parser=single_parser, description='Enter two integers')
#     def get(self):
#         args = single_parser.parse_args()
#         n1 = args.n
#         m1 = args.m
#         r = summation(n1, m1)
#         return {'add': r}

@application.route('/')
def hello_world():
    doc_ref = db.collection(u'ds23_flask').document(u'last_time_accessed')
    doc_ref.set({
        u'time': time.time(),
    })
    # print('About to print users')
    # users_ref = db.collection(u'users')
    # docs = users_ref.get()
    # for doc in docs:
    #     print(u'{} => {}'.format(doc.id, doc.to_dict()))
    # return "hi"
    # r = args.radius # request.args.get('r', type=int)
    # h = args.height # request.args.get('h', type=int)
    # return jsonify({cylinder_volume(r, h)})

@ns.route('/prediction')
class CNNPrediction(Resource):
    """Uploads your data to the CNN"""
    @api.doc(parser=single_parser, description='Upload an mnist image')
    def post(self):
        doc_ref = db.collection(u'ds23_flask').document(u'last_time_accessed')
        doc_ref.set({
            u'time': time.time(),
        })
        # print('About to print users')
        # users_ref = db.collection(u'users')
        # docs = users_ref.get()
        # for doc in docs:
        #     print(u'{} => {}'.format(doc.id, doc.to_dict()))
        args = single_parser.parse_args()
        image_file = args.file
        # print("File is:", image_file.filename if image_file else "no name")
        image_file.save('image.png')
        img = Image.open('image.png')
        image_red = img.resize((28, 28))
        image = img_to_array(image_red)
        print(image.shape)
        x = image.reshape(1, 28, 28, 1)
        x = x/255
        # This is not good, because this code implies that the model will be
        # loaded each and every time a new request comes in.
        # model = load_model('my_model.h5')
        with graph.as_default():
            out = model.predict(x)
        print(out[0])
        print(np.argmax(out[0]))
        r = np.argmax(out[0])
        doc_ref = db.collection(u'ds23_flask').document(str(time.time()))
        doc_ref.set({
            u'time': time.time(),
            u'prediction': str(r),
            u'file_name': image_file.filename if image_file else "no name"
        })
        return {'prediction': str(r)}


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Calculate volume of a Cylinder')
    # parser.add_argument('-r', '-radius', type=int, required=True)
    # parser.add_argument('-H', '-height', type=int, required=True)
    # args = parser.parse_args()
    application.run(host='0.0.0.0', port=9000)
