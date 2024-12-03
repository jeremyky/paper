# TODO List

## 11/14/2024

### Documentation Tasks
- [ ] Create comprehensive documentation explaining the project and methodology
- [ ] Prepare slideshow presentation covering:
  - [ ] Project overview
  - [ ] Methodology
  - [ ] Results
  - [ ] Future work

### Technical Tasks
- [ ] Run AlexNet on expanded image dataset
  - [ ] Generate similarity matrix for 2000+ images
  - [ ] Implement hierarchical clustering algorithm
  - [ ] Develop selection method to choose 100 representative images
    - [ ] Use distance metrics to identify images furthest from each other
    - [ ] Ensure selected images cover diverse characteristics
  - [ ] Validate selection results
  - [ ] Document the selection process and results

### Success Criteria
- Complete documentation and slides
- Successfully run AlexNet on new dataset
- Implement clustering algorithm
- Select 100 representative images from 2000+ dataset
- Verify selected images are sufficiently diverse based on distance metrics

## 11/14/2024 MEETING

- Analyze Unique Objects Dataset
- Run AlexNet on Unique Objects Dataset
- Cluster Unique Objects Dataset by using AlexNet features, first take 100 images using the CNN (distance metric as far as possible)
- Can compare w/ a random selection of 100 images to determine if the algorithm is effective at clustering by max distance (do statistical test / monte carlo)
- second cluster the 100 images into 20 clusters with 5 images per cluster (want the clusters to be as diverse as possible) <-- within THIS cluster we want the images to be as different as possible ... each group should have max varability... CLUSTERS THEMSELVES CAN LOOK SIMILAR
- possible methods of approaching this
- take 5 clusters then take 1 random image from each cluster, and then from the second cluster take the firstest from the first chosen image, then for next take the max of two distances etc. etc. ( 4 deoys 1 target per cluster )
- or take 5 clusters and take 1 random image from each cluster, etc. ( dont end up w 5 images that look similar to eachother per cluster)
- furthermore, out of the 2400 images we want to get 100, but some of them that are as different as possible may not be interpretable, so take 110 then remove 10 that are not interpretable

- after we have this we could go to do sbert... FOR LATER

- send docuemntation of ide and python and psychopy installation and new trigger sending via email



# 11 /21 /2024 MEETING

validate the grouping method by comparing it against random selection monte carlo simulation by comparing the average distances amoungst the images. 
double check to verify the algorithm is working
when selecting the 100 most diverse images, perhaps calculate the average distance between all images (computationally expensive) or take a random sample from 100 images (dont get distances of same image)
using the maxmin clustering approach, because we do it sequentially, check to see if the average distances per cluster is at a certain threshold or otherwise iteratively go back and recalculate clusters or swap ... or figure out some global method to do this

for the choosing clustering algorithm and monte carlo verification, output a similarity matrix for both so we can be able to compare images between the two

find the two closest images of the 100 images that were chosen randomly within the algorithm, and within the random selection. compare these. find the distance values that are the samllest between two images in a set. find this for each. find miniumum of each monte carlo instead of min



clean up analysis to verify the algorithm is working
i.e. max of 1000 iterations, mean avg dists of random, min dist of random, min dist of algorithm, max dist of random, max dist of algorithm, etc.
show proportions, important outputs to validate method.
output them cleanly in the end, for us to be able to put it in a report


do the random groups of 5 and plot, etc.


---

plot mean distances amongst the 20 clusters and plot them in a graph which each point representing a cluster. this will help us determine if the first few groups better diversity than the last few groups.

do the same but for the random selection groups of 5.