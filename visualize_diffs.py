from musicdiff import diff, DetailLevel, Visualization

#Visualization.INSERTED_COLOR = "green"
#Visualization.DELETED_COLOR = "red"
#Visualization.CHANGED_COLOR = "yellow"

diff("gt.krn", "prediction.krn", out_path1="Marked_piano.pdf", out_path2="Marked2.pdf", visualize_diffs=True, detail=DetailLevel.AllObjectsWithStyle)