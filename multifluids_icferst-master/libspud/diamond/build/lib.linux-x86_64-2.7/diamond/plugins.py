import os
import os.path
import sys
import thread
import traceback

import gtk.gdk

import debug

plugins = []

class PluginDetails(object):
  def __init__(self, applies, name, cb):
    self.applies = applies
    self.name = name
    self.cb = cb

  def matches(self, xpath):
    try:
      return self.applies(xpath)
    except Exception:
      debug.deprint("Warning: plugin %s raised an exception in matching function." % self.name, 0)
      return False

  def execute(self, xml, xpath):
    thread.start_new_thread(self.cb, (xml, xpath))

def register_plugin(applies, name, cb):
  global plugins
  p = PluginDetails(applies, name, cb)
  plugins.append(p)

def configure_plugins(suffix):
  homedir = os.path.expanduser('~')
  dirs = [os.path.join(homedir, ".diamond", "plugins", suffix),
      "/home/jiaye/Documents/ACSE9/multifluids_icferst-master/libspud/../share/diamond/plugins/" + suffix]
  if sys.platform != "win32" and sys.platform != "win64":
    dirs.append("/etc/diamond/plugins/" + suffix)

  for dir in dirs:
    sys.path.insert(0, dir)
    try:
      for file in os.listdir(dir):
        module_name, ext = os.path.splitext(file)
        if ext == ".py":
          try:
            debug.deprint("Attempting to import " + module_name, 1)
            module = __import__(module_name)
          except:
            debug.deprint("Plugin raised an exception:", 0)
            tb = traceback.format_exception(sys.exc_info()[0] ,sys.exc_info()[1], sys.exc_info()[2])
            tb_msg = ""
            for tbline in tb:
              tb_msg += tbline
            debug.deprint(tb_msg, 0)
    except OSError:
      pass

def cb_decorator(f):
  def wrapper(*args, **kwargs):
    gtk.gdk.threads_enter()

    try:
      f(*args, **kwargs)
    except:
      debug.deprint("Plugin raised an exception:", 0)
      tb = traceback.format_exception(sys.exc_info()[0] ,sys.exc_info()[1], sys.exc_info()[2])
      tb_msg = ""
      for tbline in tb:
        tb_msg += tbline
      debug.deprint(tb_msg, 0)

    gtk.gdk.threads_leave()
  return wrapper
