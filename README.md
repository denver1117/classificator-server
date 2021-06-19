<div align="center">
  <img src="https://github.com/denver1117/classificator/blob/master/doc/logo/main_logo.png"><br>
</div>

-----------------

# classificator-server : API and UI front-end service for the [classificator](https://github.com/denver1117/classificator) package

### About
The classificator-server provides a front-end to the [classificator](https://github.com/denver1117/classificator) python package.  The API can easily be built and served
on any linux server.  The API includes a web based UI which can be accessed from the browser using the public DNS of the host.  The UI
has extra features that augment the classificator package both before and after training.  The web server was designed and tested with Amazon EC2 in mind as the linux server of choice where organizations or individuals can quickly spin up servers of any size, build the classificator-server, and expose the web based UI to an entire network or to a single IP address.  Full instructions for spinning up a server are [here](https://github.com/denver1117/classificator-server/blob/master/serve/build/README.md).

### Features
- Easy build and install
  - Build web server and fully functional web based UI with a single command
  - Includes cleanup daemon to purge files on the server and periodically restart
- More features for the end user
  - Use HTML forms to build configuration from the UI rather than generating a JSON object
  - Infer fields from input data
  - View input data in the UI
  - View summary of input data in the UI
  - View full feature records and trained model predictions on holdout set
  - Predictor feature to make realtime predictions with a trained model after training
- Education
  - Hyperlinks to documentation for education 
  - Suggestions for tuning models for education
- Organization
  - Name and track model runs and train in parallel to others
  - View persistent log files and validation statistics through the UI

### Installation

Source Code: https://github.com/denver1117/classificator-server <br>

Clone the repo and follow the build instructions:

```
git clone https://github.com/denver1117/classificator-server.git
cd classificator-server
sh build/build_apache.sh
```

### Build

The build automatically builds the web server and serve the application:
- Install [apache web server](https://httpd.apache.org/)
- Build appropriate directories  
- Build the classificator endpoint (`/var/www/html/classificator`)
- Serve the `run_classificator.py` flask API at the classificator endpoint

### Authors

Evan Harris - Initial work began at [Return Path](https://returnpath.com/)

### License

This project is licensed under the MIT License - see the LICENSE file for details
