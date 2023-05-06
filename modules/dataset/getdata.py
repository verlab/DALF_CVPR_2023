import os, tqdm

def retrieve(data_path, reduced = False):
    #Set reduced to true to only downlad first two datasets as a sample
    datasets_links = {
            #"Alamo":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Alamo.tar", ## VALIDATION
            #"EllisIsland":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Ellis_Island.tar", ## TEST
            "MadridMetropolis":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Madrid_Metropolis.tar",  ## ALL BELOW TRAIN
            "MontrealNotreDame":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Montreal_Notre_Dame.tar",
            "NYC_Library":"http://landmark.cs.cornell.edu/projects/1dsfm/images.NYC_Library.tar",
            "PiazzadelPopolo":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Piazza_del_Popolo.tar",
            "Piccadilly":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Piccadilly.tar",
            "RomanForum":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Roman_Forum.tar",
            "TowerofLondon":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Tower_of_London.tar",
            "Trafalgar":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Trafalgar.tar",
            "UnionSquare":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Union_Square.tar",
            "ViennaCathedral":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Vienna_Cathedral.tar",
            "Yorkminster":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Yorkminster.tar",
            "Gendarmenmarkt":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Gendarmenmarkt.tar",
            }

    downloaded = 0
    os.makedirs(data_path, exist_ok=True)
    for key, item in tqdm.tqdm( datasets_links.items() ):
            if not os.path.isdir(data_path + '/' + key):
                    print("Downloading " + key )
                    if os.system("wget -nc "+ item + " -O " + data_path + '/' + key + ".tar") or \
                    os.system("tar -xv --skip-old-files -f " + data_path + '/' + key + ".tar -C " + data_path ):
                        raise RuntimeError('Failed to parse dataset.')
                    else:
                        downloaded += 1
                        if reduced and downloaded > 2:
                            break
                        # os.system("rm downloads/" + key+".tar") ### If you want to save space uncomment this line to remove the tar files
