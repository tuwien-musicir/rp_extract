
from IPython.display import HTML
import numpy as np

soundcloud_iframe = '<iframe width="100%" height="150" scrolling="no" frameborder="no" \
                         src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/{0}&amp;\
                         auto_play=false&amp;hide_related=true&amp;show_comments=false&amp;show_user=false&amp;\
                         show_reposts=false&amp;visual=true"></iframe>'

class SoundcloudTracklist(list):
    
    
    
    def _repr_html_(self):
        html = ["<table width='90%' style='border:none'>"]
        for row in self:
            html.append("<tr style='border:none'>")
            html.append("<td style='border:none'>{0}</td>".format(soundcloud_iframe.format(row)))
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)


class compareSimilarityResults(list):
    
    
    def _repr_html_(self):
        
        self = np.asarray(self).T.tolist()
        
        html = ["<table width='100%' style='border:none'>"]
        
        for row in self:
            html.append("<tr style='border:none'>")
            
            for col in row:
                html.append("<td style='border:none'>{0}</td>".format(soundcloud_iframe.format(col)))
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)