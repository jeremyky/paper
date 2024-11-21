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

## 11/21/2024 MEETING

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