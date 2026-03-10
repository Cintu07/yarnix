# Yarnix

context engine built on top of [helix](https://github.com/Cintu07/helix).

processes raw text and generates coherent language. 2 million parameters. runs on cpu.

## how it works

yarnix uses helix's phase-rotation memory as its backbone. instead of one clock speed for all neurons, yarnix runs 4 groups of neurons at different speeds simultaneously:

- ultra fast (persistence 0.5) - tracks the last few characters
- fast (0.8) - tracks sentence structure
- slow (0.95) - tracks what paragraph youre in
- ultra slow (0.999) - tracks the entire chapter

all four groups talk to each other through a cross-band mixer layer so the model knows whats happening at every timescale at the same time. this is inspired by how the brain uses gamma, beta, alpha, and theta waves for different types of processing.

## results

trained on tinyshakespeare (1 million characters), cpu only, about 6 hours.

best validation loss: **1.56**

what it generates after training:

```
MENENIUS:
Sir, I see what do you grow.
Insued me in my father's gone, he loves us to the law.

First Senator:
You have would follow the colours of the
```

```
MISIR:
I will send thee better in perfect loyalty.

KING RICHARD II:
Thanks.
```

```
GLOUCESTER:
How shall heaven is it a changeling;
And thou, with the little and the hearing of that words:
The sights of death, devisings are as many d
```

## how to run

```bash
python get_data.py          # downloads shakespeare
python language_model.py    # trains the model
```

needs pytorch and numpy. thats it.

## files

- `yarnix_cell.py` - the core architecture (YarnixCellV4 with multi-clock bands)
- `language_model.py` - training and generation
- `config.py` - settings
- `get_data.py` - dataset download

## author

pavan kalyan ([@Cintu07](https://github.com/Cintu07))
