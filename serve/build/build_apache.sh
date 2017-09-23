DIRPATH="../classificator"

if [ "$1" != "no-install" ]
then
    sudo /usr/bin/python -m pip install -r $DIRPATH/requirements.txt
    sudo apt-get update
    sudo apt-get -y install apache2
    sudo apt-get -y install libapache2-mod-wsgi
fi
sudo rm -rf /var/www/html/classificator
sudo mkdir /var/www/html/classificator
sudo cp -r $DIRPATH/* /var/www/html/classificator/  
sudo rm -rf /tmp/classificator
sudo mkdir /tmp/classificator
sudo chmod 777 /tmp/classificator
sudo chmod 777 /var/www/html/classificator
sudo chmod 777 /var/www/html/classificator/tmp
sudo chmod 777 /var/www/html/classificator/logs
sudo chmod 777 /var/www/html/classificator/file_uploads
sudo chmod 777 /var/www/html/classificator/configs
sudo chmod 777 /tmp
sudo cp $DIRPATH/build/apache_config.txt /etc/apache2/sites-enabled/000-default.conf
sudo apachectl restart
sudo rm -rf /var/log/apache2/error.log
python $DIRPATH/cleanup/cleanup_daemon.py stop
python $DIRPATH/cleanup/cleanup_daemon.py start
