#!/usr/bin/env python                                                                                                                                                                                          

#    This file is part of Diamond.
#
#    Diamond is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Diamond is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Diamond.  If not, see <http://www.gnu.org/licenses/>.

from lxml import etree
import os
import os.path
import debug
import sys
import copy
import urllib2

def preprocess(schemafile):
  p = etree.XMLParser(remove_comments=True)
  ns = 'http://relaxng.org/ns/structure/1.0'

  if 'http' in schemafile:
    schemafile_handle = urllib2.urlopen(schemafile)
  else:
    schemafile_handle = open(schemafile)
  
  try:
  	tree = etree.parse(schemafile_handle, p)
  except Exception:
  	debug.deprint("Error: %s is not a valid Relax NG schema" % schemafile, 0)
  	sys.exit(1)

  #
  # deal with include
  #
  includes = tree.xpath('/t:grammar//t:include', namespaces={'t': ns})

  for include in includes:
    include_parent = include.getparent()
    include_index = list(include_parent).index(include)

    # find the file
    file = None
    filename = include.attrib["href"]
    possible_files = [os.path.join(os.path.dirname(schemafile), filename), filename]
    possible_files.append(os.path.join("/home/jiaye/Documents/ACSE9/multifluids_icferst-master/libspud/../share/spud", filename))
    possible_files.append(os.path.join(os.path.dirname(__file__) + "/../../schema", filename))

    for possible_file in possible_files:
      try:
        if 'http' in possible_file:
          file = urllib2.urlopen(possible_file)
        else:
          file = open(possible_file)
        break
      except IOError:
        debug.deprint("IOError when searching for included file " + filename, 1)

    if file is None:
      debug.deprint("Error: could not locate included file %s" % filename, 0)
      debug.deprint("Path: %s" % possible_files)
      sys.exit(1)

    # parse the included xml file and steal all the nodes
    include_tree = etree.parse(file, p)
    nodes_to_take = include_tree.xpath('/t:grammar/*', namespaces={'t': ns})

    # here's where the magic happens:
    for node in nodes_to_take:
      include_parent.insert(include_index, copy.deepcopy(node))

    # now delete the include:
    include_parent.remove(include)

  grammar_list = tree.xpath('/t:grammar', namespaces={'t': ns})
  
  # If the .rnc didn't include a start = prefix, then no valid
  # grammar tag will be present. Let the user know.
  
  if len(grammar_list) == 0:
  		debug.deprint("Error: No grammar tag present in schema.", 0)
	  	sys.exit(1)

  grammar = grammar_list[0]

  defines = {}
  define_nodes = tree.xpath('/t:grammar//t:define', namespaces={'t': ns})

  #
  # deal with combine="interleave"
  #

  # first, fetch all the plain definitions
  for define in define_nodes:
    if "combine" not in define.attrib:
      name = define.attrib["name"]
      defines[name] = define

  # now look for interleaves with those
  for define in define_nodes:
    if "combine" in define.attrib and define.attrib["combine"] == "interleave":
      name = define.attrib["name"]
      if name not in defines:
        defines[name] = define
      else:
        matching_defn = defines[name]
        for child in define:
          matching_defn.append(copy.deepcopy(child))
  
  #
  # deal with combine="choice"
  #
  combine_names = []
  for define in define_nodes:
    if "combine" in define.attrib and define.attrib["combine"] == "choice":
      name = define.attrib["name"]
      combine_names.append(name)

  combine_names = list(set(combine_names))
  for name in combine_names:
    xpath = tree.xpath('/t:grammar//t:define[@name="%s"]' % name, namespaces={'t': ns})
    choices = []
    for node in xpath:
      choices = choices + list(node)
    define = etree.Element("define")
    define.attrib["name"] = name
    choice = etree.Element("choice")
    define.append(choice)
    for x in choices:
      choice.append(x)
    defines[name] = define

  # delete all the define nodes from the xml
  for define in define_nodes:
    parent = define.getparent()
    parent.remove(define)
  
  # add the modified defines back to the grammar
  for define in defines.values():
    grammar.append(define)

  return etree.tostring(tree, xml_declaration=True, encoding='utf-8', pretty_print=True)

if __name__ == "__main__":
  import sys
  schemafile = sys.argv[1]
  newfile = schemafile.replace(".rng", ".pp.rng")
  f = open(newfile, "w")
  f.write(preprocess(schemafile))
