What's **_GLANCE_**?

**_GLANCE_** (acronym for Gravitational Lensing Authenticator using Non-modelled Cross-ocrrelation Exploration) is a tool to find gravitationally lensed images. 
Whether it's a signle distorted image or a multi-image system, **_GLANCE_** finds them all.

Why **_GLANCE_**?

**_GLANCE_** does not assume any astrophysical approximations and assumptions of objects. It works unmodelled. It relies on the fact that if there's feature, 
that's physical, the feature must be present to data from different independent observations.

When **_GLANCE_**?

The idea orginated in 2024, when we found out that most of the lensing detection algorithms for lensed gravitational waves searches relied on astrophysics, which 
we don't know much about. A true detection cannot be made when models are overwhelming. That's when **_GLANCE_** came out.

Who's **_GLANCE_**?

Well, that's not a valid question, but you can ask who we are. I am Aniruddha and I'm from the <data|theory> universe lab led by Prof. Suvodip Mukherjee, 
from Tata Institute of Fundamental Research, Mumbai, the funding body of this work.

How's **_GLANCE_**?

Well, **_GLANCE_** is doing well. We are planning to launch a v1.0 soon. Stay tuned!

Is there something called **_$\mu$-GLANCE_**?

Yes, that's **_GLANCE_**'s junior sibling. While **_GLANCE_** itself looks for strongly lensed signals, **_$\mu$-GLANCE_** (called micro-GLANCE) looks for 
microlensed distorted signals.

---------------------------------------

**INSTALLATION :**

Step I (git cloning, so that files move to your local):

git clone https://github.com/AniruddhaCIndia/glance.git

Step II (Find installer, it's in the same folder):

cd glance

Step III (Install glance):

pip install -e .
(Any changes ypu make to **_GLANCE_** are accounted without reinstalling it again.)

Step IV (Update installation):

cd glance
git pull origin main

----------------------------------------

If you're interested in contributing to the development of **_GLANCE_**, please get in touch: aniruddha.chakraborty@tifr.res.in

