"""
Python daemon class
"""

import sys, os, time, atexit
import argparse
import subprocess
from signal import SIGTERM
 
class ClassificatorCleanup:
        """
        A generic daemon class.
       
        Usage: subclass the Daemon class and override the run() method
        """
        def __init__(self, pidfile, stdin='/dev/null', stdout='/dev/null', stderr='/dev/null', sleep_time=43200, purge="5+"):
                self.stdin = stdin
                self.stdout = stdout
                self.stderr = stderr
                self.pidfile = pidfile
                self.sleep_time = int(sleep_time)
                self.purge = str(purge)
                self.locs = [
                        "/tmp/classificator/*", 
                        "/var/www/html/classificator/configs/*.json"
                    ]
       
        def daemonize(self):
                """
                do the UNIX double-fork magic, see Stevens' "Advanced
                Programming in the UNIX Environment" for details (ISBN 0201563177)
                http://www.erlenstar.demon.co.uk/unix/faq_2.html#SEC16
                """
                try:
                        pid = os.fork()
                        if pid > 0:
                                # exit first parent
                                sys.exit(0)
                except OSError, e:
                        sys.stderr.write("fork #1 failed: %d (%s)\n" % (e.errno, e.strerror))
                        sys.exit(1)
       
                # decouple from parent environment
                os.chdir("/")
                os.setsid()
                os.umask(0)
       
                # do second fork
                try:
                        pid = os.fork()
                        if pid > 0:
                                # exit from second parent
                                sys.exit(0)
                except OSError, e:
                        sys.stderr.write("fork #2 failed: %d (%s)\n" % (e.errno, e.strerror))
                        sys.exit(1)
       
                # redirect standard file descriptors
                sys.stdout.flush()
                sys.stderr.flush()
                si = file(self.stdin, 'r')
                so = file(self.stdout, 'a+')
                se = file(self.stderr, 'a+', 0)
                os.dup2(si.fileno(), sys.stdin.fileno())
                os.dup2(so.fileno(), sys.stdout.fileno())
                os.dup2(se.fileno(), sys.stderr.fileno())
       
                # write pidfile
                atexit.register(self.delpid)
                pid = str(os.getpid())
                file(self.pidfile,'w+').write("%s\n" % pid)
       
        def delpid(self):
                os.remove(self.pidfile)
 
        def start(self):
                """
                Start the daemon
                """
                # Check for a pidfile to see if the daemon already runs
                try:
                        pf = file(self.pidfile,'r')
                        pid = int(pf.read().strip())
                        pf.close()
                except IOError:
                        pid = None
       
                if pid:
                        message = "pidfile %s already exist. Daemon already running?\n"
                        sys.stderr.write(message % self.pidfile)
                        sys.exit(1)
               
                # Start the daemon
                self.daemonize()
                self.run()
 
        def stop(self):
                """
                Stop the daemon
                """
                # Get the pid from the pidfile
                try:
                        pf = file(self.pidfile,'r')
                        pid = int(pf.read().strip())
                        pf.close()
                except IOError:
                        pid = None
       
                if not pid:
                        message = "pidfile %s does not exist. Daemon not running?\n"
                        sys.stderr.write(message % self.pidfile)
                        return # not an error in a restart
 
                # Try killing the daemon process       
                try:
                        while 1:
                                os.kill(pid, SIGTERM)
                                time.sleep(0.1)
                except OSError, err:
                        err = str(err)
                        if err.find("No such process") > 0:
                                if os.path.exists(self.pidfile):
                                        os.remove(self.pidfile)
                        else:
                                print str(err)
                                sys.exit(1)
 
        def restart(self):
                """
                Restart the daemon
                """
                self.stop()
                self.start()
 
        def run(self):
                while True:
                        # sleep
                        time.sleep(self.sleep_time)

                        # perform cleanup
                        subprocess.Popen("sudo apachectl restart", shell=True)
                        for loc in self.locs:
                                subprocess.Popen(
                                    ("sudo find " + loc + 
                                     " -mtime " + self.purge + 
                                     " -exec rm -rf {} \;"), 
                                    shell=True)
                        done = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='classificator cleanup daemon')
    parser.add_argument('mode', nargs='?')
    parser.add_argument('-t', '--time',
                        dest='time',
                        default='43200',
                        help='sleep time (in seconds) between cleanup runs')
    parser.add_argument('-p', '--purge',
                        dest='purge',
                        default='7+',
                        help='purge time')
    args = parser.parse_args()
    daemon = ClassificatorCleanup(
        '/tmp/classificator-daemon.pid',
        sleep_time=args.time,
        purge=args.purge)
    if args.mode:
        if args.mode == 'start':
            daemon.start()
        elif args.mode == 'stop':
            daemon.stop()
        elif args.mode == 'restart':
            daemon.restart()
        else:
            print "cleanup_daemon: unknown mode: {0}".format(args.mode)
            sys.exit(2)
        sys.exit(0)
    else:
        print "cleanup_daemon: must enter a positional mode argument"
        sys.exit(2)

