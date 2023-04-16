import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--Classifier", type=str,required=True, help = "Classifier path. From more info visit GitHub page.")

    args = parser.parse_args((' '.join(sys.argv[1:])).split())
    
    if args.Classifier:
        process(args.Classifier)
    
def process(classifier:str):
    from hand_detection_interface import GUI
    g = GUI(classifier)
    g.main_gui()
    
if __name__ == '__main__':
    main()





