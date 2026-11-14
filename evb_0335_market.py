"""Stochastic EVB-0335 asset quote evaluated over 50 ticks."""

import matplotlib.pyplot as plt

from syphen import (
    DependentExpressionElement,
    SemanticAttribute,
    SemanticValueType,
    System,
    SystemState,
    SystemicSpace,
    ValueConstraints,
    VariableElement,
)
from sypron import NormalDistribution, RandomVariableElement, SypronStateModel


systemic_space = SystemicSpace()
market = System(systemic_space, label="EVB-0335 market")
market_state = SystemState(systemic_space)
market.suscribe_systemic_description(market_state.identificator.id)

asset_archetype = market.semantics.define_archetype_from_instance(
    "Equity asset",
    market.identificator.id,
    tags=("market asset",),
)
market.semantics.add_attribute(
    asset_archetype.id,
    SemanticAttribute(
        name="ticker",
        value_type=SemanticValueType.STRING,
        default="EVB-0335",
    ),
)
market.semantics.add_tag(market.identificator.id, "asset>equity")
market.semantics.add_base_dimension("currency")
usd = market.semantics.define_unit("US dollar", "USD", {"currency": 1})

price = RandomVariableElement(
    systemic_space,
    "price",
    NormalDistribution(mean=100.05, standard_deviation=1.0),
    initial_value=100.0,
    constraints=ValueConstraints(minimum=0.0),
)
market_state.add_element(price.identificator.id)
market.semantics.bind_state_variable(market_state, price.identificator.id, usd.id)

price_history_01 = VariableElement(
    systemic_space, "price_history_01", initial_value=100.0,
    constraints=ValueConstraints(minimum=0.0),
)
market_state.add_element(price_history_01.identificator.id)
market.semantics.bind_state_variable(market_state, price_history_01.identificator.id, usd.id)

price_history_02 = VariableElement(
    systemic_space, "price_history_02", initial_value=100.0,
    constraints=ValueConstraints(minimum=0.0),
)
market_state.add_element(price_history_02.identificator.id)
market.semantics.bind_state_variable(market_state, price_history_02.identificator.id, usd.id)

price_history_03 = VariableElement(
    systemic_space, "price_history_03", initial_value=100.0,
    constraints=ValueConstraints(minimum=0.0),
)
market_state.add_element(price_history_03.identificator.id)
market.semantics.bind_state_variable(market_state, price_history_03.identificator.id, usd.id)

price_history_04 = VariableElement(
    systemic_space, "price_history_04", initial_value=100.0,
    constraints=ValueConstraints(minimum=0.0),
)
market_state.add_element(price_history_04.identificator.id)
market.semantics.bind_state_variable(market_state, price_history_04.identificator.id, usd.id)

distribution_expression = DependentExpressionElement(
    systemic_space,
    "distribution_expression",
    price.symbol * 1.0005,
)
market_state.add_element(distribution_expression.identificator.id)

moving_average_5 = DependentExpressionElement(
    systemic_space,
    "moving_average_5",
    (
        price.symbol
        + price_history_01.symbol
        + price_history_02.symbol
        + price_history_03.symbol
        + price_history_04.symbol
    )
    / 5,
)
market_state.add_element(moving_average_5.identificator.id)
market_state.compile_state(
    {
        distribution_expression.symbol: distribution_expression.expression,
        moving_average_5.symbol: moving_average_5.expression,
    }
)

market_model = SypronStateModel(market_state)
price_values = []
moving_average_values = []

realization_01 = market_model.sample()
evaluation_01 = market_model.evaluate(realization_01)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_01[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_01[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 01: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_02 = market_model.sample()
evaluation_02 = market_model.evaluate(realization_02)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_02[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_02[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 02: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_03 = market_model.sample()
evaluation_03 = market_model.evaluate(realization_03)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_03[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_03[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 03: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_04 = market_model.sample()
evaluation_04 = market_model.evaluate(realization_04)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_04[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_04[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 04: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_05 = market_model.sample()
evaluation_05 = market_model.evaluate(realization_05)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_05[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_05[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 05: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_06 = market_model.sample()
evaluation_06 = market_model.evaluate(realization_06)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_06[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_06[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 06: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_07 = market_model.sample()
evaluation_07 = market_model.evaluate(realization_07)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_07[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_07[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 07: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_08 = market_model.sample()
evaluation_08 = market_model.evaluate(realization_08)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_08[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_08[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 08: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_09 = market_model.sample()
evaluation_09 = market_model.evaluate(realization_09)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_09[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_09[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 09: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_10 = market_model.sample()
evaluation_10 = market_model.evaluate(realization_10)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_10[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_10[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 10: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_11 = market_model.sample()
evaluation_11 = market_model.evaluate(realization_11)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_11[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_11[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 11: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_12 = market_model.sample()
evaluation_12 = market_model.evaluate(realization_12)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_12[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_12[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 12: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_13 = market_model.sample()
evaluation_13 = market_model.evaluate(realization_13)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_13[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_13[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 13: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_14 = market_model.sample()
evaluation_14 = market_model.evaluate(realization_14)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_14[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_14[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 14: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_15 = market_model.sample()
evaluation_15 = market_model.evaluate(realization_15)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_15[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_15[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 15: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_16 = market_model.sample()
evaluation_16 = market_model.evaluate(realization_16)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_16[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_16[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 16: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_17 = market_model.sample()
evaluation_17 = market_model.evaluate(realization_17)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_17[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_17[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 17: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_18 = market_model.sample()
evaluation_18 = market_model.evaluate(realization_18)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_18[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_18[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 18: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_19 = market_model.sample()
evaluation_19 = market_model.evaluate(realization_19)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_19[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_19[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 19: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_20 = market_model.sample()
evaluation_20 = market_model.evaluate(realization_20)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_20[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_20[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 20: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_21 = market_model.sample()
evaluation_21 = market_model.evaluate(realization_21)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_21[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_21[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 21: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_22 = market_model.sample()
evaluation_22 = market_model.evaluate(realization_22)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_22[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_22[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 22: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_23 = market_model.sample()
evaluation_23 = market_model.evaluate(realization_23)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_23[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_23[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 23: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_24 = market_model.sample()
evaluation_24 = market_model.evaluate(realization_24)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_24[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_24[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 24: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_25 = market_model.sample()
evaluation_25 = market_model.evaluate(realization_25)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_25[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_25[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 25: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_26 = market_model.sample()
evaluation_26 = market_model.evaluate(realization_26)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_26[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_26[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 26: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_27 = market_model.sample()
evaluation_27 = market_model.evaluate(realization_27)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_27[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_27[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 27: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_28 = market_model.sample()
evaluation_28 = market_model.evaluate(realization_28)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_28[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_28[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 28: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_29 = market_model.sample()
evaluation_29 = market_model.evaluate(realization_29)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_29[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_29[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 29: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_30 = market_model.sample()
evaluation_30 = market_model.evaluate(realization_30)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_30[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_30[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 30: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_31 = market_model.sample()
evaluation_31 = market_model.evaluate(realization_31)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_31[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_31[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 31: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_32 = market_model.sample()
evaluation_32 = market_model.evaluate(realization_32)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_32[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_32[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 32: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_33 = market_model.sample()
evaluation_33 = market_model.evaluate(realization_33)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_33[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_33[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 33: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_34 = market_model.sample()
evaluation_34 = market_model.evaluate(realization_34)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_34[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_34[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 34: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_35 = market_model.sample()
evaluation_35 = market_model.evaluate(realization_35)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_35[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_35[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 35: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_36 = market_model.sample()
evaluation_36 = market_model.evaluate(realization_36)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_36[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_36[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 36: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_37 = market_model.sample()
evaluation_37 = market_model.evaluate(realization_37)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_37[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_37[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 37: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_38 = market_model.sample()
evaluation_38 = market_model.evaluate(realization_38)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_38[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_38[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 38: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_39 = market_model.sample()
evaluation_39 = market_model.evaluate(realization_39)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_39[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_39[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 39: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_40 = market_model.sample()
evaluation_40 = market_model.evaluate(realization_40)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_40[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_40[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 40: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_41 = market_model.sample()
evaluation_41 = market_model.evaluate(realization_41)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_41[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_41[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 41: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_42 = market_model.sample()
evaluation_42 = market_model.evaluate(realization_42)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_42[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_42[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 42: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_43 = market_model.sample()
evaluation_43 = market_model.evaluate(realization_43)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_43[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_43[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 43: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_44 = market_model.sample()
evaluation_44 = market_model.evaluate(realization_44)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_44[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_44[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 44: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_45 = market_model.sample()
evaluation_45 = market_model.evaluate(realization_45)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_45[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_45[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 45: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_46 = market_model.sample()
evaluation_46 = market_model.evaluate(realization_46)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_46[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_46[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 46: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_47 = market_model.sample()
evaluation_47 = market_model.evaluate(realization_47)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_47[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_47[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 47: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_48 = market_model.sample()
evaluation_48 = market_model.evaluate(realization_48)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_48[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_48[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 48: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_49 = market_model.sample()
evaluation_49 = market_model.evaluate(realization_49)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_49[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_49[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 49: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

realization_50 = market_model.sample()
evaluation_50 = market_model.evaluate(realization_50)
price_values.append(market_state.read_value(price.identificator.id))
moving_average_values.append(float(evaluation_50[moving_average_5.symbol]))
price.distribution = NormalDistribution(float(evaluation_50[distribution_expression.symbol]), 1.0)
market_state.set_values({price_history_01.identificator.id: price_values[-1], price_history_02.identificator.id: market_state.read_value(price_history_01.identificator.id), price_history_03.identificator.id: market_state.read_value(price_history_02.identificator.id), price_history_04.identificator.id: market_state.read_value(price_history_03.identificator.id)})
print(f"Tick 50: price={price_values[-1]:.4f} USD, moving_average_5={moving_average_values[-1]:.4f} USD")

plt.plot(price_values, color = "blue")
plt.plot(moving_average_values, color = "red")
plt.show()