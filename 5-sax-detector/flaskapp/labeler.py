import flask
from flask import request, render_template
from pymongo import MongoClient
import os

client = MongoClient("mongodb://{}:{}@{}/kojak".format(
        os.environ['mdbUN'],
        os.environ['mdbPW'],
        os.environ['mdbIP']
    )
)
kdb = client.kojak
# coll = kdb.flask_test
coll = kdb.test_songs

app = flask.Flask(__name__)


def query_mongo():
    """
    Pulls a random unlabeled chunk ID from the MongoDB and creates a link to 
    the MP3.
    ---
    IN
    None
    OUT
    Rendered page
    """
    
    sample = coll.aggregate([{"$match": {"labeled": {"$exists": False}}},
                             {"$sample": {"size": 1}}
                            ])
    try:
        # serves up the main page with a link to the sampled MP3
        doc = next(sample)
        chunk_id = doc['chunk_id']
        print("\n<-- Serving page with following sample:")
        print("Chunk:", chunk_id)
        print("Song:", doc['song_name'])
        link = "http://davidluthermusic.com/kojak/audio/" + chunk_id + ".mp3"
        # pass the chunk_id up as well and store it as a var
        return render_template("labeler.html", 
                               mp3_link = link,
                               served_id = int(chunk_id))
    except:
        # serve up the page
        print("\n<-- No songs to label, serving 'finished' page")
        return render_template("finished.html")


def check_mc(mc):
    """
    Returns proper status or error message based on modified count of MongoDB
    query.
    ---
    IN
    mc: result.modified_count (int)
    OUT
    status: message based on modified count (str)
    """

    if mc == 1:
        status = " * Sample modified in DB"
    elif mc == 0:
        status = " * ERROR: No samples modified in DB"
    else:
        status = " * ERROR: Modified count of {}".format(mc)

    return status


@app.route("/")
def home():
    return render_template("intro.html")    


@app.route("/labeler")
def labeler():
    return query_mongo()


@app.route("/skip_sample", methods=["POST"])
def skip_sample():
    """
    Labels the sample as labeled/useless.
    """

    data = flask.request.get_json()
    chunk_id = str(data['chunk_id']).rjust(6, '0')
    print("\n--> Data received:", data)
    print(" * Skip this sample:", chunk_id) 
    
    # records sample as labeled/skipped in Mongo
    try:
        result = coll.update_one({"chunk_id": chunk_id},
                                 {"$set": {"labeled": True,
                                           "skipped": True
                                          }
                                 }
                                )
        print(check_mc(result.modified_count))
    except:
        print(" * ERROR: Check database connection")
    
    return flask.jsonify({'status': 'skipped'})


@app.route("/label_sample", methods=["POST"])
def label_sample():
    """
    Labels the sample with instruments it contains.
    """

    data = flask.request.get_json()
    chunk_id = str(data['chunk_id']).rjust(6, '0')
    print("\n--> Data received:", data)
    print(" * Log data for this sample:", chunk_id)

    # records sample labels and status in DB
    try:
        result = coll.update_one({"chunk_id": chunk_id},
                                 {"$set": {"labeled": True,
                                           "skipped": False,
                                           "sax": data['sax'],
                                           "piano": data['pno'],
                                           "vocals": data['vox']
                                          }
                                 }
                                )
        print(check_mc(result.modified_count))
    except:
        print(" * ERROR: Check database connection")

    return flask.jsonify({'status': 'logged'})


# Start app server on port 8000
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
