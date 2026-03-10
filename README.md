# Yarnix

character level language model built on top of [helix](https://github.com/Cintu07/helix).

trains on raw text, learns to write shakespeare. 2 million parameters. runs on cpu.

## how it works

yarnix stores memory as angles on a circle instead of numbers. numbers decay over time. angles dont. thats the whole trick.

the model has 4 groups of neurons running at different speeds:
- fast ones (persistence 0.5) track the last few characters
- medium ones (0.8) track sentence structure  
- slow ones (0.95) track what paragraph youre in
- ultra slow ones (0.999) track the whole chapter

all four groups talk to each other through a mixer layer so the model knows whats happening at every timescale simultaneously.

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

## how to run

```bash
python get_data.py          # downloads shakespeare
python language_model.py    # trains the model
```

needs pytorch and numpy. thats it.

## files

- `yarnix_cell.py` - the core architecture
- `language_model.py` - training and generation
- `config.py` - settings
- `get_data.py` - dataset download

## author

pavan kalyan ([@Cintu07](https://github.com/Cintu07))
