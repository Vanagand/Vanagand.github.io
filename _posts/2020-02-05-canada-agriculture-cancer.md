---
layout: post
title: Cancer Incidence and Agriculture in Canada
subtitle: Is there corrolation?
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
tags: [cancer, agriculture, canada, territories]
bigimg: /img/cdn_bigimg_banner.jpg
comments: true
---
## Introduction

Cancer can be one of those current topics that is either personal or foreign to people. Even with the strides made in the fields of medicine and research, it still burdens the lives of many. With this article I aim to provide an attempt to describe the correlation between the amount of land area dedicated to agriculture compared against the incidence of cancer in the many provinces and territories of Canada.

The goal of this study is to learn the ropes of data science and analysis with sprinkled visualization. Land area dedicated to agriculture was chosen to support the plausible hypothesis that there is a weak, but positive correlation. Farmers and people that reside close to farmland are usually exposed to an array of things such as engine fuel, organic and inorganic dust, fumes, heat, noise, zoonotic viruses, and chemicals. It could be said that the weight of the correlation describes the severity of the exposures mentioned above.

## Overview of the territory

<img id="cdn_census_area" src="/img/cdn_census_area_plot.png" alt="graph">{:height="200%" width="200%"}

First and foremost, let’s explore the agricultural sector of the Canadian provinces. As shown above, only a few provinces stands above the rest in term of land dedicated to agriculture – Saskatchewan and Alberta, which together with Manitoba accounts for the great western plains, or great prairies of Canada. Only a relatively small proportion of the land area is suitable for farming which varies considerably by region, which is tied to the difference in soils, climates, and topography. Saskatchewan accounts for about 40 per cent of all farmland area while Alberta and Manitoba account for 31 per cent and 11 per cent respectively. Having this information is crucial as it provides a clear bracket for agricultural weight in the country – the 3 aforementioned account for about more than a cumulative 80 per cent of all arable land in Canada. Some provinces were omitted due to soil, climates, and topography, only a relatively small area of land is suitable for farming. 

Another thing to denote is that the overall land area has been in a slow decline for decades. This will play a part in the following graph as the overall trend of cancer incidents across Canada has been on the rise.

## Overview of the problem

<img id="cdn_census_cancer" src="/img/cdn_census_cancer_plot.png" alt="graph">{:height="200%" width="200%"}

There are a couple things we can take away from the graph above, all of which points toward a positive mean line over time meaning that the incidence of cancer rates in Canada are increasing. It also seems that the provinces with the highest rate of cancer incidences are the ones with the least amount of land dedicated to agriculture. 

## Conclusion

<img id="cdn_census_cancer" src="/img/cdn_cancer_area_heat.png" alt="graph">{:height="200%" width="200%"}

When calculating the correlation between the two variables of arable land area and the incidence of cancer in Canada we find that these two actually have a strong negative correlation. We can look further and accept our initial hypothesis that there would atleast be a weak but positive correlation between the two. 

The study above paints a very wide description of an issue that remains at the heart of this country. It is unclear if the incidence of cancer rates can be attributed to the presence of agriculture, but the idea is not dismissed. More variables and observations would be required in order to draw a more conclusive argument.

It is also worth noting that during the past few years, efforts were made to reduce the use of both pesticide and fertilizers and some legislations were put in place in order to reduce or limit the purchase of those products. Homeowners are now left with the option of approved bio products that have less of an impact on the environment. Taking these factors into account, it is difficult to draw a solid conclusion regarding the direct impact of agriculture in regards of cancer incidents on the population.

If cancer occurs more prominently among a certain population, then it is important to understand this information in order to provide more efficient preventive methods. It is clear that there are many other variables in place. Our first pass shows that some of the provinces with the highest rate of cancer have smaller agricultural sectors with hard-to-reach markets, and that some of the largest agricultural sectors, in contrast, have lower cancer incidence per population. 

Stay healthy friends, 

Link to notebook: [Notebook](https://github.com/Vanagand/DS-Unit-1-Build/blob/master/DS_Unit_1_Build.ipynb)
