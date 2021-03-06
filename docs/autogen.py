from keras_autodoc import DocumentationGenerator
import keras_autodoc
import shutil

client_methods = keras_autodoc.get_methods('flybrainlab.Client.Client')
ordered_methods = ['flybrainlab.Client.Client',
                    'flybrainlab.Client.Client.__init__',
                    'flybrainlab.Client.Client.tryComms',
                    'flybrainlab.Client.Client.executeNLPquery',
                    'flybrainlab.Client.Client.executeNAquery']
client_methods_final = ordered_methods + [i for i in client_methods if i not in ordered_methods]

pages = {'client.md': client_methods_final,
         'widgets.md': ['flybrainlab.widget.CallbackManager','flybrainlab.widget.WidgetManager','flybrainlab.Client.MetaClient']}

doc_generator = DocumentationGenerator(pages)
doc_generator.generate('./sources')
shutil.copyfile('index.md','sources/index.md')