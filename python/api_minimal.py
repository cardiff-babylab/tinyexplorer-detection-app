from flask import Flask
from flask_cors import CORS
from graphene import ObjectType, String, Schema, Float, List, Boolean
from flask_graphql import GraphQLView
from calc import calc as real_calc
import argparse
import os
import threading
import json

#
# Minimal API without heavy ML dependencies
#

class Query(ObjectType):

    awake = String(description="Awake")
    def resolve_awake(self, args):
        return "Awake"

    exit = String(description="Exit", signingkey=String(required=True))
    def resolve_exit(self, info, signingkey):
        if signingkey != apiSigningKey:
            return
        os._exit(0)
        return

    hello = String(description="Hello", signingkey=String(required=True))
    def resolve_hello(self, info, signingkey):
        if signingkey != apiSigningKey:
            return "invalid signature"
        return "World"
    
    calc = String(description="Calculator", signingkey=String(required=True), math=String(required=True))
    def resolve_calc(self, info, signingkey, math):
        """based on the input text, return the int result"""
        if signingkey != apiSigningKey:
            return "invalid signature"
        try:
            return real_calc(math)
        except Exception:
            return 0.0
    
    echo = String(description="Echo", signingkey=String(required=True), text=String(required=True))
    def resolve_echo(self, info, signingkey, text):
        if signingkey != apiSigningKey:
            return "invalid signature"
        """echo any text"""
        return text

view_func = GraphQLView.as_view("graphql", schema=Schema(query=Query), graphiql=True)

parser = argparse.ArgumentParser()
parser.add_argument("--apiport", type=int, default=5000)
parser.add_argument("--signingkey", type=str, default="")
args = parser.parse_args()

apiSigningKey = args.signingkey

app = Flask(__name__)
app.add_url_rule("/graphql/", view_func=view_func)
app.add_url_rule("/graphiql/", view_func=view_func) # for compatibility with other samples
CORS(app) # Allows all domains to access the flask server via CORS

if __name__ == "__main__":
    app.run(port=args.apiport)