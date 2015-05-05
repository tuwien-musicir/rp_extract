import subprocess


class Wrapper(object):
    '''
    classdocs
    '''


    def __init__(self, bin_path):
        
        self.binary_path = bin_path
        
    def call_binary(self, cmd):
        
        
        
        try:
            
            cmd.insert(0,self.binary_path)
            
            # open pipe to shell and execute command
            process = subprocess.Popen(cmd, 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE)
            
            # store cmd-line output
            std_out, std_err = process.communicate()
            
            # convert output to utf-8
            std_out = unicode(std_out, errors='ignore')
            std_err = unicode(std_err, errors='ignore')

        except UnicodeDecodeError as e:
            pass
            #todo handle this
        
        return std_out, std_err
    
    