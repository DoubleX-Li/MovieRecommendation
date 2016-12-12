# -*- coding: UTF-8 –*-

import time, sys, cheerypy, os
from paste.translogger import TransLogger
from app import create_app
from pyspark import SparkContext, SparkConf


def init_spark_context():
    # 载入spark context
    conf = SparkConf().setAppName("movie_recommendation-server")
    # IMPORTANT: pass aditional Python modules to each worker
    sc = SparkContext(conf=conf, pyFiles=['engine.py', 'app.py'])

    return sc


def run_server(app):
    # Enable WSGI access logging via Paste
    app_logged = TransLogger(app)

    # Mount the WSGI callable object (app) on the root directory
    cherrypy.tree.graft(app_logged, '/')

    # Set the configuration of the web server
    cherrypy.config.update({
        'engine.autoreload.on': True,
        'log.screen': True,
        'server.socket_port': 8888,
        'server.socket_host': '0.0.0.0'
    })

    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()


if __name__ == "__main__":
    # 初始化spark context并载入libraries
    sc = init_spark_context()
    dataset_path = os.path.join('datasets', 'ml-latest')
    app = create_app(sc, dataset_path)

    # 开启web服务器
    run_server(app)