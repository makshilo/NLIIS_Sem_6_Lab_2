from nltk import pos_tag, word_tokenize, RegexpParser
from nltk.tree.prettyprinter import TreePrettyPrinter
from nltk.draw import TreeView


def generate_tree(sentence):
    # Find all parts of speech in above sentence
    tagged = pos_tag(word_tokenize(sentence))

    # Extract all parts of speech from any text
    chunker = RegexpParser("""
                           NP: {<DT>?<JJ>*<NN>}    #To extract Noun Phrases
                           P: {<IN>}               #To extract Prepositions
                           V: {<V.*>}              #To extract Verbs
                           PP: {<p> <NP>}          #To extract Prepositional Phrases
                           VP: {<V> <NP|PP>*}      #To extract Verb Phrases
                           """)

    # Print all parts of speech in above sentence
    output = chunker.parse(tagged)
    return output


def draw_tree(tree, filename):
    # Draw tree
    TreeView(tree)._cframe.print_to_file(filename + '.ps')


def tree_to_text(tree):
    # Convert tree to text
    return TreePrettyPrinter(tree).text()

