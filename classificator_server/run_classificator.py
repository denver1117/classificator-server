"""
Classificator flask API serving as UI backend and standalone API
for the classify python module
"""

from classificator.classify import Classificator, load_sklearn_dataset
from classificator import models
from flask import Flask, abort, render_template, request, jsonify, redirect, url_for
from flask.ext.navigation import Navigation
from werkzeug.utils import secure_filename

import ast
import datetime
import traceback
import json
import pprint
import os, glob
import pandas as pd
import numpy as np
import time
import boto3
import pickle
import random
import subprocess

# Get working directory to assemble paths
try:
    cwd = os.path.dirname(__file__)
except:
    cwd = os.getcwd()
uploaddir = "{0}/file_uploads".format(cwd)
logdir = "/tmp"
tmpdir = "/tmp"
out_default = "/tmp/classificator"
python_path = "/usr/bin/python"
html_path = "/var/www/html/classificator"

# For file uploads
UPLOAD_FOLDER = "{0}/file_uploads".format(html_path)
ALLOWED_EXTENSIONS = set(["tsv", "csv"])

# Instantiate app and nav bar
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
nav = Navigation(app)
nav.Bar('top', [
    nav.Item('Home', 'index'),
    nav.Item('Configure Project', 'configure', {'page': 1}),
    nav.Item('Project Hub', 'projects', {'page': 2}),
    nav.Item('About', 'about', {'page': 3}),
    nav.Item('The Pipeline', 'pipeline', {'page': 4}),
    nav.Item('API', 'api_ref', {'page': 5}),
    nav.Item('Install', 'install', {'page': 5}),
    nav.Item('Contact', 'contact', {'page': 7}),
])
processes = {}

#########################
### Utility Functions ###
#########################

def load_model(loc):
    """ Load and return pickled model object from S3 or local """

    # If location is in S3, copy to local, then unpickle
    to_delete = False
    if "s3" in loc:
        tmp_loc = "{0}/tmp_file_{1}.obj".format(tmpdir, random.randint(1,1000))
        s3 = boto3.client('s3')
        bucket = loc.split("/")[2]
        key = "/".join(loc.split("/")[3:])
        with open(tmp_loc, "wb") as data:
            s3.download_fileobj(bucket, key, data)
        loc = tmp_loc
        to_delete = True

    with open(loc, "rb") as f:
        model = pickle.load(f)
    if to_delete:
        os.remove(tmp_loc)
    return model

def kill_thread(run_name):
    """ Kill thread with the given name """

    if run_name in processes.keys():
        try:
            processes[run_name].kill()
        except:
            pass
    else:
        raise Exception("Could not kill run")

def is_alive(run_name):
    """ Is process alive for given run_name """

    if run_name in processes.keys():
        polled = processes[run_name].poll() 
        if polled is None:
            return "Yes"
        else:
            return "No"
    else:
        return "Maybe"

def merge_configs(old, new):
    """ Merge old and new configurations """

    for key in old.keys():
        if key not in new.keys():
            new[key] = old[key]
    return new

def infer_method(name, is_numeric, unique_pct):
    """ Infer method for the given field """

    for kword in ["label", "target", "result"]:
        if kword in name.lower():
            return "label"
    if is_numeric:
        return "numeric"
    elif unique_pct > 0.01:
        return "vectorize"
    else:
        return "encode"

def try_func(clf):
    """
    Try to run the Classificator.choose_model method 
    writing results to the log file
    """

    try:
        clf.choose_model()
    except Exception as e:
        # Write the traceback to log
        with open("{0}".format(clf.log_name), "a") as f:
            now = datetime.datetime.now()
            f.write("{0} - Runtime Error\n".format(now))
            tb = traceback.format_exc()
            f.write(tb)

def threaded(clf):
    """ Run Classificator method on a separate thread """

    run_name = clf.config["data_specs"]["run_name"]
    killed = kill_thread(run_name)
    p = Process(target=try_func, args=(clf,))
    p.start()
    app.jobs[run_name] = p

def eval_literal(x):
    """ Attempt literal eval of input """

    try:
        return ast.literal_eval(x)
    except:
        pass
    return x

def get_config_loc(run_name):
    """ Get name of config file """

    return "{0}/configs/{1}.json".format(cwd, run_name)

def load_config(run_name):
    """ Load configuration json """

    with open(get_config_loc(run_name)) as data_file:    
        data = json.load(data_file)
    return data

def write_config(run_name, data, mode="w"):
    """ Write configuration json """

    with open(get_config_loc(run_name), mode) as outfile:
        json.dump(data, outfile)


def parse_field_config(result):
    """ Parse form output for feature, group and label methods """

    keys = [x for x in result.keys() if "__methods__" in x]

    # Handle feature columns
    result["data_specs__feature_columns"] = [
        x.split("__")[-1] for x in keys 
        if result[x].replace('"', "") not in ["group", "label", "none"]]
    result["data_specs__feature_methods"] = [
        result[x].replace('"', "") for x in keys 
        if result[x].replace('"', "") not in ["group", "label", "none"]]
    args = np.argsort(result["data_specs__feature_columns"])
    result["data_specs__feature_columns"] = list(np.array(result["data_specs__feature_columns"])[args])
    result["data_specs__feature_methods"] = list(np.array(result["data_specs__feature_methods"])[args])
    result["data_specs__prediction_defaults"] = [
        result["data_specs__prediction_defaults"][col]
        for col in result["data_specs__feature_columns"]]
    a = []
    a_ = []
    b = []
    b_ = []
    c = []
    c_ = []
    for col in result["data_specs__feature_columns"]:
        if "data_specs__{0}__a".format(col) in result.keys():
            a.append(col)
            a_.append(result["data_specs__feature_methods"][result["data_specs__feature_columns"].index(col)])
        if "data_specs__{0}__b".format(col) in result.keys():
            b.append(col)
            b_.append(result["data_specs__feature_methods"][result["data_specs__feature_columns"].index(col)])
        if "data_specs__{0}__c".format(col) in result.keys():
            c.append(col)
            c_.append(result["data_specs__feature_methods"][result["data_specs__feature_columns"].index(col)])
    feat_cols = []
    feat_methods = []
    for lst in [[a, a_],[b, b_],[c, c_]]:
        if len(lst[0]) > 0:
            feat_cols.append(lst[0])
            feat_methods.append(lst[1])
    if len(feat_cols) > 1:
        result["data_specs__feature_columns"] = feat_cols
        result["data_specs__feature_methods"] = feat_methods

    # Handle group and label columns
    result["data_specs__group_columns"] = [
        x.split("__")[-1] for x in keys 
        if result[x].replace('"', "") == "group"]
    result["data_specs__label_column"] = [
        x.split("__")[-1] for x in keys 
        if result[x].replace('"', "") == "label"][0]
    return result  

def parse_form(result_dct):
    """ 
    Parse the form output to match the expected format of the
    Classificator config input 
    """

    if result_dct["data_specs__loc"].split("/")[0] == "uploads":
        result_dct["data_specs__loc"] = "{0}/{1}".format(
            uploaddir, 
            "/".join(result_dct["data_specs__loc"].split("/")[1:]))
    config = {}
    if "selector__grid" in result_dct.keys():
        result_dct["selector__grid"] = (
            "{'alpha': " + parse_list(result_dct["selector__grid"]) + "}")
    if "pre_processors__methods" in result_dct.keys():
        preproc_map = dict(
            scaler=["Standard Scaler"], none=[], 
            max=["Max Scaler"], normalizer=["Normalizer"])
        result_dct["pre_processors__methods"] = str([
            preproc_map[x.strip()] for x in 
            parse_list(result_dct["pre_processors__methods"]).replace("[", "").replace("]", "").split(",")]) 
    for key in result_dct.keys():
        val = eval_literal(result_dct[key])
        if val != '':
            splitted = key.split("__")
            if splitted[0] not in config.keys():
                config[splitted[0]] = {}
            if len(splitted) == 2:
                config[splitted[0]][splitted[1]] = val
            elif len(splitted) == 3:
                if splitted[1] not in config[splitted[0]].keys():
                    config[splitted[0]][splitted[1]] = {}
                config[splitted[0]][splitted[1]][splitted[2]] = val
    return config

def handle_loc(loc, run_name):
    """ Handles file location by appending run name and clearing output directory """

    # generate new filename
    new_dir =  (
        "{0}{1}".format(loc, run_name)
        if loc[-1] == "/" else
        "{0}/{1}".format(loc, run_name))

    # if S3 delete all keys else create local directory or 
    # delete conents of existing
    if "s3://" in loc.lower():
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(loc.lower().split("s3://")[1].split("/")[0])
        prefix = "{0}/".format("/".join(new_dir.lower().split("s3://")[1].split("/")[1:]))
        for obj in bucket.objects.filter(Prefix=prefix):
            s3.Object(bucket.name, obj.key).delete()
    elif not os.path.exists(new_dir):
        os.makedirs(new_dir)
    else:
        filelist = glob.glob("{0}/*".format(new_dir))
        for f in filelist:
            os.remove(f)
    return new_dir

def infer_sep(path, return_sep=False):
    """ Infer file separator """

    for sep in [",", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if return_sep:
                return df, sep
            else:
                return df
        except:
            pass

def parse_list(val):
    """ Parse a list of input strings """

    val = (
        val
        .replace("[", "").replace("]", "")
        .replace("(", "").replace(")", "")
        .replace("{", "").replace("}", "")
        .strip()
        .lower())
    if "," not in val:
        val = ",".join(val.split())
    val = "[{0}]".format(val)
    return val

def allowed_file(filename):
    """ Is file extension allowed for upload"""

    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_validation_file(clf_name, filedir):
    """ Load CV results file from output location """

    filename = "{0}.tsv".format(clf_name.replace(" ", "_"))
    try:
        return pd.read_csv("{0}/{1}".format(filedir, filename), sep="\t")
    except:
        pass

def predict(result):
    """ Make prediction according to posted data input """


    config = load_config(result["run_name"])
    model = load_model(
        "{0}/{1}".format(
            config["data_specs__out_loc"],
            config["data_specs__model_name"]))

    # Assemble input features
    features = [result[x] for x in model.features]

    # Attempt prediction
    try:
        prediction = model.predict_proba([features])
        arr = map(
            lambda x: round(x, 4),
            model.predict_proba([features])[0])
        dct = dict(zip(model.classes_, arr))
        top = model.classes_[np.argmax(arr)]
        return ([
            "-- Top Class --", top, "-- Class Probabilities --"] +
            ["{0}: {1}".format(key, value) for key, value in dct.iteritems()])
    except Exception as e:
        prediction = None
        return "Error", str(traceback.format_exc())

#####################
### API Endpoints ###
#####################

@app.route('/')
def index():
    """ Render homepage """

    return render_template('index.html')

@app.route('/configure')
def configure():
   """ Render configuration input form """

   return render_template('configure_load.html')

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

@app.route('/result', methods = ['POST'])
def result():
    """ 
    Parse form output, kick off model run and 
    show realtime logging
    """

    result = request.form
    config = load_config(result["data_specs__run_name"])
    for key in config.keys():
        if "classifiers__" in key:
            del config[key]
    to_write = merge_configs(config, result.to_dict())
    write_config(result["data_specs__run_name"], to_write, mode="w")
    config = parse_form(to_write)    
    tmp_config_loc = "{0}/{1}.json".format(
        tmpdir, result["data_specs__run_name"])
    with open(tmp_config_loc, 'w') as outfile:
        json.dump(config, outfile)
    log_name = "{0}/{1}.txt".format(
        logdir, result["data_specs__run_name"])
    with open(log_name, "w"):
            pass
    try:
        processes[result["data_specs__run_name"]].kill()
    except:
        pass
    processes[result["data_specs__run_name"]] = subprocess.Popen([
        python_path, "{0}/classify_runner.py".format(html_path), 
        "--config", tmp_config_loc])
    return redirect(url_for('train_landing'))

@app.route('/train_landing', methods = ['GET'])
def train_landing():
    """ Landing page for training  """

    return render_template("train_landing.html")

@app.route('/logs', methods = ['GET'])
def log():
    """ Render log file for given run_name argument """

    run_name = request.args.get("run_name")
    log_name = "{0}/{1}.txt".format(logdir, run_name)
    f = open(log_name, "r")
    result = [x for x in f.readlines()]
    f.close()
    finished = "no"
    return render_template(
        "logs.html", result=result, 
        run_name=run_name, finished=finished)

@app.route('/kill_run', methods = ['GET'])
def kill_run():
    """ Kill model run """

    killed = kill_thread(request.args.get("run_name"))
    return render_template("killed.html")

@app.route('/about')
def about():
    """ Render about page """

    return render_template("about.html")


@app.route('/install')
def install():
    """ Render install page """

    return render_template("install.html")

@app.route('/contact')
def contact():
    """ Render contact page """

    return render_template("contact.html")

@app.route('/pipeline')
def pipeline():
    """ Render pipeline page """

    return render_template("pipeline.html")

@app.route('/api_ref')
def api_ref():
    """ Render API reference page """

    return render_template("api_ref.html")

@app.route('/projects')
def projects():
    """ Projects dashboard """

    projects = [
        x.split("/")[-1].replace(".json", "") for x in 
        glob.glob("{0}/configs/*.json".format(cwd))]
    is_alive_list = [is_alive(x) for x in projects]
    rng = range(len(projects))
  
    return render_template(
        "projects.html", projects=projects, 
        is_alive_list=is_alive_list, rng=rng)

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    """ Upload file from client """

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template("file_upload.html")
    return render_template("file_upload.html")

@app.route('/dataset', methods=['POST'])
def dataset():
    """ Load, validate and view dataset """

    result = request.form.to_dict()

    # Initial configuration repairs
    if result["data_specs__loc"].split("/")[0].lower() == "uploads":
        result["data_specs__loc"] = "{0}/{1}".format(UPLOAD_FOLDER, result["data_specs__loc"].split("/")[1])
    if ("data_specs__out_loc" not in result.keys()) or (result["data_specs__out_loc"].strip() == ""):
        result["data_specs__out_loc"] = out_default
    result["data_specs__out_loc"] = handle_loc(result["data_specs__out_loc"], result["data_specs__run_name"])
    
    output = {}
    # Load dataset
    if result["data_specs__loc"].lower() in models.datasets.keys():
        df = load_sklearn_dataset(result["data_specs__loc"].lower())
    else:
        try:
            df, sep = infer_sep(
                result["data_specs__loc"], 
                return_sep=True)
            result["data_specs__sep"]= sep
        except:
            tb = traceback.format_exc().split("\n")
            return render_template("data_load_failure.html", result=tb)

    # Generate info dataframe
    info = df.describe(include="all", percentiles=[])

    # Save median values for prediction defaults
    if "top" in info.index and "mean" in info.index:
        prediction_defaults = [(info[col]["top"] if ~pd.isnull(info[col]["top"]) else str(round(info[col]["mean"],2))) for col in info.columns]
    elif "mean" in info.index:
        prediction_defaults = [str(round(info[col]["mean"],2)) for col in info.columns]
    else:
        prediction_defaults = [info[col]["top"] for col in info.columns]
    result["data_specs__prediction_defaults"] = dict(zip(info.columns, prediction_defaults))
    write_config(result["data_specs__run_name"], result, mode="w")

    # Infer field types
    if len(df) > 1000:
        df = df.sample(1000, random_state=10)
    cols = list(info.columns)
    is_numeric = ~np.array([pd.isnull(info[col]["mean"]) for col in info.columns])
    unique_pct = []
    for index in range(len(cols)):
        if is_numeric[index]:
            unique_pct.append(np.nan)    
        else:
            unique_pct.append(info[cols[index]]["unique"] / float(info[cols[index]]["count"]))
    info = info.reset_index().fillna("")
    default_values = [infer_method(cols[x], is_numeric[x], unique_pct[x]) for x in range(len(cols))]

    # Prepare selection defaults for html template
    select_options = []
    select_visible = ["none", "vectorize", "encode", "numeric", "group", "label"]
    for index in range(len(default_values)):
        select_options.append(['value="{0}"{1}'.format(x, " selected" if default_values[index] == x else "") for x in select_visible])
    output["select_options"] = select_options
    output["select_visible"] = select_visible
    output["select_index"] = range(len(select_visible))

    # Write data to dictionary for html template
    output["columns"] = cols
    output["index"] = range(len(df.columns))
    output["values"] = map(
            lambda x: [''.join(i for i in str(y) if ord(i) < 128) for y in x],
            df.values)
    output["info_columns"] = info.columns
    output["info_values"] = map(lambda x: [''.join(i for i in str(y) if ord(i) < 128) for y in x], info.values)
    return render_template(
        "data_load_success.html", result=output, 
        run_name=result["data_specs__run_name"])

@app.route('/configure_training', methods=['POST'])
def configure_training():
    """ Configure training meta parameters """

    result = request.form.to_dict()
    run_name = result["data_specs__run_name"]
    config = load_config(run_name)
    to_write = merge_configs(config, result)
    to_write = parse_field_config(to_write)
    write_config(run_name, to_write, mode="w")
    return render_template(
        "configure_training.html", run_name=run_name)

@app.route('/view_results', methods=['POST', 'GET'])
def view_results():
    """ View training run validation stats and sample predictions """

    # Max sample size
    n = 500

    # Format configuration
    if request.method == "POST":
        result = request.form.to_dict()
        run_name = result["data_specs__run_name"]
    elif request.method == "GET":
        run_name = request.args.get("run_name")
    config = load_config(run_name)
    config = parse_form(config)
    out_loc = config["data_specs"]["out_loc"]
    out_loc = out_loc[:-1] if out_loc[-1] == "/" else out_loc

    # Load CV results data
    output = {}
    reports_suffixes = ["main", "A", "B", "C"]
    for clf_name in config["classifiers"].keys():
        output[clf_name] = {}
        for suffix in reports_suffixes:
            tmpDf = load_validation_file("{0}_{1}".format(clf_name.lower(), suffix), out_loc)
            if tmpDf is not None:
                tmpDf = tmpDf.sort_values("rank_test_score")
                tmpDf = tmpDf[[c for c in tmpDf.columns if ("param_" not in c) and (c != "rank_test_score")]]
                for col in tmpDf.columns:
                    if ("_time" in col) or ("_score" in col):
                        tmpDf.loc[:, col] = tmpDf[col].apply(lambda x: round(x,4))
                output[clf_name][suffix] = [list(tmpDf.columns), tmpDf.values]

    # Load hold out data
    hold_out_df = pd.read_csv("{0}/{1}".format(out_loc, "hold_out_data.tsv"), sep="\t")
    crosstabs = pd.crosstab(hold_out_df["label_true"], hold_out_df["label_pred"]).reset_index()
    crosstabs = [list(crosstabs.columns), map(
            lambda x: [''.join(i for i in str(y) if ord(i) < 128) for y in x],
            crosstabs.values)]
    correct = hold_out_df.loc[hold_out_df["label_true"] == hold_out_df["label_pred"],:].sort_values("label_true")
    if len(correct) >= n:
        correct = correct.sample(n, random_state=10).sort_values("label_true")
    preds = [list(correct.columns), map(
            lambda x: [''.join(i for i in str(y) if ord(i) < 128) for y in x],
            correct.values)]
    incorrect = hold_out_df.loc[hold_out_df["label_true"] != hold_out_df["label_pred"],:].sort_values("label_true")
    if len(incorrect) >= n:
        incorrect = incorrect.sample(n, random_state=10).sort_values("label_true")
    errors = [list(incorrect.columns), map(
            lambda x: [''.join(i for i in str(y) if ord(i) < 128) for y in x],
            incorrect.values)]

    # Load classification report
    try:
        clf_rpt = pd.read_csv("{0}/{1}".format(out_loc, "clf_report_main.tsv"), sep="\t") 
    except:
        clf_rpt = pd.read_csv("{0}/{1}".format(out_loc, "clf_report_combined.tsv"), sep="\t")
    for index in range(len(clf_rpt))[:-1]:
        if pd.isnull(clf_rpt["class"].values[index]):
            try:
                clf_rpt.loc[index, "class"] = crosstabs[0][index + 1]
            except:
                pass
    clf_report = [list(clf_rpt.columns), clf_rpt.values]
    return render_template(
        "view_output.html", result=output, preds=preds, 
        errors=errors, clf_report=clf_report, 
        run_name=run_name, crosstabs=crosstabs)

@app.route('/predictor', methods = ['GET', 'POST'])
def predictor():
    """ Predictor UI rendering """

    # Parse input data
    if request.method == "POST":
        result = request.form
    elif request.method == "GET":
        result = request.args
    result = result.to_dict()
    config = load_config(result["run_name"])
    if isinstance(config["data_specs__feature_columns"][0], list):
        config["data_specs__feature_columns"] = list(np.sort(list(set([item for sublist in config["data_specs__feature_columns"] for item in sublist]))))
    config["idx"] = range(len(config["data_specs__feature_columns"]))
    if "method" in result.keys():
        prediction_data = predict(result)
    else:
        prediction_data = [""]
    return render_template(
        "predictor.html", config=config, 
        prediction_data=prediction_data)

### Standalone API endpoint

@app.route('/classificate', methods = ['POST'])
def classificate():
    """ Post config to endpoint to kick off model run """

    input = request.get_json(force=True)
    clf = Classificator(config=input)
    try:
        clf.choose_model()
        status = "SUCCESS"
        tb = ""
    except Exception as e:
        status = "FAILURE"
        tb = traceback.format_exc()

    output = [
        {
         'input': input, 
         'status': status,
         'traceback': tb
        }
    ]
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("8080")
    )

