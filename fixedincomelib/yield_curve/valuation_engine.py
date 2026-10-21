"""Analytic valuation of a signed fixed-accrual cashflow on an IFR model."""

import numpy as np

from fixedincomelib.model import ModelType
from fixedincomelib.product import ProductFixedAccruedCashflow
from fixedincomelib.valuation import (
    AnalyticValParam, FundingIndexParameter, ValuationEngineProduct,
    ValuationEngineProductRegistry,
)


class ValuationEngineProductFixedAccrued(ValuationEngineProduct):
    def __init__(self, model, valuation_parameters_collection, product, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        # TODO 
        if not isinstance(product, ProductFixedAccruedCashflow):
            raise TypeError("ValuationEngineProductFixedAccrued requires a ProductFixedAccruedCashflow")
        if model.model_type != ModelType.YIELD_CURVE.to_string():
            raise TypeError("ValuationEngineProductFixedAccrued requires a YIELD_CURVE model")

        # Select the funding (discounting) curve for the product's currency.
        if not valuation_parameters_collection.has_vp_type(FundingIndexParameter._vp_type):
            raise ValueError(
                f"A '{FundingIndexParameter._vp_type}' valuation parameter is required to "
                "choose the discounting curve."
            )
        funding_parameter: FundingIndexParameter = (
            valuation_parameters_collection.get_vp_from_build_method_collection(
                FundingIndexParameter._vp_type
            )
        )
        self.funding_index_ = funding_parameter.get_funding_index(product.currency)
        # Fail early if the model does not contain that funding curve.
        self.model.retrieve_model_component(self.funding_index_)

        # Contractual quantities that do not depend on the model.
        self.payment_date_ = product.payment_date
        self.cash_amount_ = product.notional * product.accrued  # C = N * alpha (signed)

    # ------------------------------------------------------------------ helpers
    @property
    def funding_index(self):
        return self.funding_index_

    @property
    def payment_date(self):
        return self.payment_date_

    @property
    def cash_amount(self) -> float:
        """Signed undiscounted cash amount ``C = N * alpha``."""
        return self.cash_amount_

    def _is_future_payment(self) -> bool:
        return self.payment_date_ > self.value_date

    def _is_same_day_payment(self) -> bool:
        return self.payment_date_ == self.value_date

    def calculate_value(self) -> None:
        cash_amount = self.cash_amount_
        if self._is_future_payment():
            # PV = C * D_funding(T_p); nothing settles today.
            discount_factor = self.model.discount_factor(
                self.funding_index_, self.payment_date_, self.product_.currency.code()
            )
            self.value_ = cash_amount * discount_factor
            self.cash_ = 0.0
        elif self._is_same_day_payment():
            # Pays today: full amount is cash, and it is included in the PV.
            self.value_ = cash_amount
            self.cash_ = cash_amount
        else:
            # Already paid: nothing left to value.
            self.value_ = 0.0
            self.cash_ = 0.0

    def calculate_risk(
        self, gradient: list[np.ndarray], scaler: float = 1.0, accumulate: bool = False,
    ) -> None:
        # TODO Write product risk using the model's DF gradient; non-future payments have zero risk.
        # Apply scaler to the new contribution; overwrite or add according to accumulate.
        if self._is_future_payment():
            # The model helper allocates the blocks and honours overwrite/accumulate.
            self.model.discount_factor_gradient_wrt_state(
                self.funding_index_, self.payment_date_, gradient,
                scaler=scaler * self.cash_amount_, accumulate=accumulate,
            )
        else:
            # Zero risk: make sure the blocks exist, and clear them when overwriting.
            self.model.resize_gradient(gradient)
            if not accumulate:
                for block in gradient:
                    block.fill(0.0)


ValuationEngineProductRegistry().register(
    (ModelType.YIELD_CURVE.to_string(), ProductFixedAccruedCashflow._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductFixedAccrued,
)
