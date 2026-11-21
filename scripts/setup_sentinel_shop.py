"""
Setup SentinelShop demo repository with realistic code and Git history
such that BugLens discovers HIGH (>=66%), MEDIUM (33-66%), and LOW (<33%) risk functions.
"""
import os
import stat
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, timezone

repo_dir = Path("data/demo_repos/sentinel_shop").resolve()

def remove_readonly(func, path, excinfo):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

# Step 1: Wipe existing git history and recreate
git_dir = repo_dir / ".git"
if git_dir.exists():
    shutil.rmtree(git_dir, onerror=remove_readonly)

orders_dir = repo_dir / "src" / "orders"
payments_dir = repo_dir / "src" / "payments"
users_dir = repo_dir / "src" / "users"
orders_dir.mkdir(parents=True, exist_ok=True)
payments_dir.mkdir(parents=True, exist_ok=True)
users_dir.mkdir(parents=True, exist_ok=True)

# Write source files
checkout_code = '''"""
SentinelShop — Order Processing & Checkout Engine
"""
from typing import Dict, List, Optional, Any
from datetime import datetime


class CheckoutError(Exception):
    pass


class InvalidCouponError(CheckoutError):
    pass


class InsufficientInventoryError(CheckoutError):
    pass


class FraudDetectionAlert(CheckoutError):
    pass


def format_order_confirmation(order_id: str, total_amount: float, customer_email: str) -> Dict[str, Any]:
    """Simple, clean formatting function (Expected: LOW RISK)."""
    return {
        "order_id": order_id,
        "total": round(total_amount, 2),
        "recipient": customer_email.strip().lower(),
        "timestamp": datetime.utcnow().isoformat(),
        "status": "CONFIRMED"
    }


def calculate_shipping_and_tax(weight_kg: float, zone: str, express: bool = False, is_member: bool = False) -> float:
    """Medium complexity shipping rate calculation (Expected: LOW RISK)."""
    if weight_kg <= 0:
        raise ValueError("Weight must be greater than zero.")
    
    base_rate = 5.0
    if zone == "DOMESTIC_METRO":
        base_rate += weight_kg * 1.5
    elif zone == "DOMESTIC_REGIONAL":
        base_rate += weight_kg * 2.5
    elif zone == "INTERNATIONAL_STANDARD":
        base_rate += weight_kg * 7.0 + 15.0
    elif zone == "INTERNATIONAL_REMOTE":
        base_rate += weight_kg * 12.0 + 30.0
    else:
        base_rate += weight_kg * 4.0

    if express:
        if zone.startswith("INTERNATIONAL"):
            base_rate *= 1.8
        else:
            base_rate *= 1.4

    if is_member and base_rate > 20.0:
        base_rate -= 5.0

    tax_rate = 0.08 if not zone.startswith("INTERNATIONAL") else 0.0
    total_shipping = base_rate * (1.0 + tax_rate)
    return round(total_shipping, 2)


def process_complex_checkout(
    cart_items: List[Dict[str, Any]],
    coupon_code: Optional[str] = None,
    customer_tier: str = "BRONZE",
    currency: str = "USD",
    user_loyalty_points: int = 0,
    shipping_address: Optional[Dict[str, str]] = None,
    tax_exempt: bool = False,
    is_gift: bool = False,
    max_discount_limit: float = 350.0,
    fraud_risk_score: float = 0.05,
    user_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Core Order Settlement Engine (Expected: HIGH RISK).
    Contains multi-warehouse inventory verification, recursive discount stacking,
    fraud velocity filtering, multi-currency conversion, and jurisdiction tax rules.
    """
    if not cart_items:
        raise CheckoutError("Cannot process checkout with an empty cart.")
    if shipping_address is None or not shipping_address.get("country"):
        raise CheckoutError("Valid shipping address is mandatory.")

    country = shipping_address.get("country", "").upper()
    state = shipping_address.get("state", "").upper()

    # Fraud & Risk Velocity Assessment
    if fraud_risk_score > 0.85:
        raise FraudDetectionAlert("Order blocked due to high-risk fraud velocity score.")
    elif fraud_risk_score > 0.60:
        if customer_tier not in ["PLATINUM", "VIP", "ENTERPRISE"]:
            raise FraudDetectionAlert("Manual KYC verification required for elevated risk score.")

    subtotal = 0.0
    total_weight = 0.0
    item_count = 0
    restricted_items = False
    hazardous_materials = False

    for item in cart_items:
        price = item.get("unit_price", 0.0)
        qty = item.get("quantity", 0)
        category = item.get("category", "general").upper()
        stock = item.get("stock_available", 0)
        item_weight = item.get("weight_kg", 0.25)

        if qty <= 0:
            raise CheckoutError(f"Invalid item quantity: {qty}")
        if price < 0.0:
            raise CheckoutError(f"Invalid negative price: {price}")
        if qty > stock:
            raise InsufficientInventoryError(f"Item {item.get('id')} out of stock.")

        if category == "HAZARDOUS":
            hazardous_materials = True
            restricted_items = True
        elif category in ["AGE_RESTRICTED", "LIQUOR", "FIREARMS"]:
            restricted_items = True

        subtotal += price * qty
        total_weight += item_weight * qty
        item_count += qty

    if restricted_items and country not in ["US", "CA", "GB", "DE", "FR", "AU"]:
        raise CheckoutError(f"Restricted items cannot be shipped internationally to {country}.")

    # Tier-based Base Loyalty Discount
    tier_discount = 0.0
    if customer_tier == "ENTERPRISE":
        tier_discount = subtotal * 0.20
    elif customer_tier in ["PLATINUM", "VIP"]:
        tier_discount = subtotal * 0.15
    elif customer_tier == "GOLD":
        tier_discount = subtotal * 0.10
    elif customer_tier == "SILVER":
        tier_discount = subtotal * 0.05
    elif customer_tier == "BRONZE":
        tier_discount = 0.0
    else:
        tier_discount = 0.0

    # Coupon Validation & Stackability Matrix
    coupon_discount = 0.0
    if coupon_code:
        code = coupon_code.strip().upper()
        if code.startswith("SUMMER"):
            if subtotal >= 100.0:
                coupon_discount = min(40.0, subtotal * 0.20)
            else:
                raise InvalidCouponError("SUMMER coupon requires minimum $100 cart value.")
        elif code.startswith("VIP"):
            if customer_tier in ["PLATINUM", "VIP", "ENTERPRISE", "GOLD"]:
                coupon_discount = subtotal * 0.25
            else:
                raise InvalidCouponError("VIP coupon reserved for Gold/Platinum/Enterprise members.")
        elif code == "FLAT50":
            if subtotal >= 200.0:
                coupon_discount = 50.0
            else:
                coupon_discount = subtotal * 0.10
        elif code == "FREESHIP":
            coupon_discount = 15.0
        elif code.startswith("PARTNER_"):
            if len(code) > 10 and subtotal >= 150.0:
                coupon_discount = subtotal * 0.12
            else:
                coupon_discount = 10.0
        else:
            raise InvalidCouponError(f"Unrecognized coupon code: {coupon_code}")

    # Loyalty Point Redemption
    point_discount = 0.0
    if user_loyalty_points > 0:
        max_usable_points = min(user_loyalty_points, int(subtotal * 20))
        if customer_tier in ["PLATINUM", "VIP", "ENTERPRISE"]:
            point_discount = (max_usable_points / 100.0) * 1.5
        elif customer_tier == "GOLD":
            point_discount = (max_usable_points / 100.0) * 1.2
        else:
            point_discount = (max_usable_points / 100.0) * 1.0

    raw_total_discount = tier_discount + coupon_discount + point_discount
    actual_discount = min(raw_total_discount, max_discount_limit, subtotal * 0.70)
    discounted_subtotal = max(0.0, subtotal - actual_discount)

    # Shipping Zone Calculation
    if country in ["US", "CA"]:
        zone = "DOMESTIC_METRO" if item_count < 10 else "DOMESTIC_REGIONAL"
    elif country in ["GB", "DE", "FR", "NL", "IT"]:
        zone = "INTERNATIONAL_STANDARD"
    else:
        zone = "INTERNATIONAL_REMOTE"

    is_premium_tier = customer_tier in ["PLATINUM", "VIP", "ENTERPRISE"]
    shipping_fee = calculate_shipping_and_tax(
        weight_kg=total_weight,
        zone=zone,
        express=is_gift,
        is_member=is_premium_tier
    )

    if hazardous_materials:
        shipping_fee += 25.0

    # Regional Tax Calculation
    tax_amount = 0.0
    if not tax_exempt:
        if country == "US":
            if state in ["CA", "NY", "WA"]:
                tax_amount = discounted_subtotal * 0.0925
            elif state in ["TX", "FL", "IL"]:
                tax_amount = discounted_subtotal * 0.0825
            elif state in ["OR", "MT", "DE", "NH", "AK"]:
                tax_amount = 0.0
            else:
                tax_amount = discounted_subtotal * 0.0650
        elif country in ["GB", "DE", "FR", "IT", "ES"]:
            if country == "DE":
                tax_amount = discounted_subtotal * 0.19
            elif country == "FR":
                tax_amount = discounted_subtotal * 0.20
            elif country == "GB":
                tax_amount = discounted_subtotal * 0.20
            elif country == "ES":
                tax_amount = discounted_subtotal * 0.21
            else:
                tax_amount = discounted_subtotal * 0.22
        elif country == "IN":
            tax_amount = discounted_subtotal * 0.18
        elif country == "AU":
            tax_amount = discounted_subtotal * 0.10
        elif country == "JP":
            tax_amount = discounted_subtotal * 0.10
        else:
            tax_amount = discounted_subtotal * 0.05

    pre_currency_total = discounted_subtotal + shipping_fee + tax_amount

    # Currency Conversion Multiplier Matrix
    rates = {
        "USD": 1.0,
        "EUR": 0.92,
        "GBP": 0.79,
        "JPY": 154.5,
        "CAD": 1.38,
        "AUD": 1.52,
        "INR": 84.3
    }

    if currency.upper() not in rates:
        raise CheckoutError(f"Unsupported settlement currency: {currency}")

    fx_rate = rates[currency.upper()]
    converted_total = pre_currency_total * fx_rate
    final_settlement = round(converted_total, 2)

    return {
        "order_id": f"ORD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{item_count}",
        "currency": currency.upper(),
        "subtotal": round(subtotal * fx_rate, 2),
        "discount_applied": round(actual_discount * fx_rate, 2),
        "shipping_fee": round(shipping_fee * fx_rate, 2),
        "tax_amount": round(tax_amount * fx_rate, 2),
        "final_total": final_settlement,
        "items_count": item_count,
        "customer_tier": customer_tier,
        "zone": zone,
        "is_gift": is_gift,
        "status": "READY_FOR_PAYMENT"
    }
'''

gateway_code = '''"""
SentinelShop — Payment Gateway & Webhook Reconciliation
"""
import hmac
import hashlib
import time
from typing import Dict, Any, Optional


def generate_transaction_receipt(txn_id: str, code: str) -> str:
    """Simple receipt generator (Expected: LOW RISK)."""
    return f"RCPT-{txn_id[:8].upper()}-{code}"


def apply_wallet_credit(balance: float, credit_amount: float, max_cap: float = 500.0) -> float:
    """Medium risk wallet adjustment (Expected: LOW RISK)."""
    if balance < 0 or credit_amount < 0:
        raise ValueError("Amounts cannot be negative.")
    if credit_amount > max_cap:
        raise ValueError(f"Credit exceeds allowed max limit: {max_cap}")
    new_balance = balance + credit_amount
    if new_balance > 5000.0:
        return 5000.0
    return round(new_balance, 2)


def reconcile_transaction_webhook(
    payload: Dict[str, Any],
    signature_header: str,
    secret_key: str,
    max_retries: int = 3,
    enforce_idempotency: bool = True
) -> Dict[str, Any]:
    """
    Webhook Reconciler & State Machine (Expected: MEDIUM RISK).
    Validates HMAC signature, checks replay attack tokens, routes through
    complex payment state transitions and dispute resolution handlers.
    """
    if not payload or not isinstance(payload, dict):
        raise ValueError("Invalid webhook payload structure.")
    if not signature_header or not secret_key:
        raise PermissionError("Missing cryptographic authentication credentials.")

    # 1. HMAC SHA256 Signature Verification
    raw_event = payload.get("event_id", "")
    timestamp = payload.get("timestamp", 0)
    current_time = int(time.time())

    if abs(current_time - timestamp) > 300:
        raise PermissionError("Webhook timestamp expired or clock skew exceeds 300s.")

    computed_sig = hmac.new(
        secret_key.encode("utf-8"),
        f"{raw_event}:{timestamp}".encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(computed_sig, signature_header):
        raise PermissionError("Cryptographic signature mismatch.")

    # 2. Idempotency Check
    idempotency_key = payload.get("idempotency_key")
    if enforce_idempotency and not idempotency_key:
        raise ValueError("Missing mandatory idempotency key.")

    event_type = payload.get("type", "unknown")
    txn_data = payload.get("data", {})
    txn_id = txn_data.get("transaction_id")
    amount = txn_data.get("amount", 0.0)
    initial_status = txn_data.get("status", "PENDING")

    if not txn_id:
        raise ValueError("Transaction ID missing in event data.")

    # 3. Multi-State Transition Processing Matrix
    final_status = initial_status
    action_taken = "NOOP"
    retry_count = 0

    while retry_count < max_retries:
        try:
            if event_type == "charge.succeeded":
                if initial_status in ["PENDING", "AUTHORIZED"]:
                    if amount > 10000.0:
                        final_status = "FLAGGED_HIGH_VALUE"
                        action_taken = "NOTIFY_FRAUD_TEAM"
                    else:
                        final_status = "SETTLED"
                        action_taken = "FULFILL_ORDER"
                else:
                    final_status = "CONFLICT_ALREADY_CLOSED"
                    action_taken = "LOG_ANOMALY"
                break
            elif event_type == "charge.failed":
                error_code = txn_data.get("failure_code", "UNKNOWN")
                if error_code in ["INSUFFICIENT_FUNDS", "EXPIRED_CARD"]:
                    final_status = "FAILED_SOFT"
                    action_taken = "PROMPT_USER_NEW_CARD"
                elif error_code in ["FRAUDULENT", "STOLEN_CARD"]:
                    final_status = "FAILED_HARD"
                    action_taken = "BLOCK_ACCOUNT"
                else:
                    final_status = "FAILED_GENERIC"
                    action_taken = "RETRY_DISPATCH"
                break
            elif event_type == "charge.refunded":
                refund_amt = txn_data.get("refund_amount", amount)
                if refund_amt >= amount:
                    final_status = "REFUNDED_FULL"
                    action_taken = "RESTOCK_INVENTORY"
                else:
                    final_status = "REFUNDED_PARTIAL"
                    action_taken = "ADJUST_LEDGER"
                break
            elif event_type == "charge.disputed":
                dispute_reason = txn_data.get("dispute_reason", "general")
                if dispute_reason == "fraudulent":
                    final_status = "DISPUTE_CRITICAL"
                    action_taken = "FREEZE_ESCROW_AND_ALERT_LEGAL"
                else:
                    final_status = "UNDER_DISPUTE"
                    action_taken = "FREEZE_ESCROW_FUNDS"
                break
            elif event_type == "subscription.renewed":
                if amount > 0.0:
                    final_status = "SUBSCRIPTION_ACTIVE"
                    action_taken = "EXTEND_PERIOD"
                else:
                    final_status = "SUBSCRIPTION_ZERO_CHARGE"
                    action_taken = "AUDIT_FREE_TIER"
                break
            else:
                final_status = "IGNORED_UNRECOGNIZED"
                action_taken = "DISCARD"
                break
        except Exception:
            retry_count += 1
            if retry_count >= max_retries:
                final_status = "ERROR_RETRY_EXHAUSTED"
                action_taken = "SEND_TO_DEAD_LETTER_QUEUE"

    return {
        "event_id": raw_event,
        "transaction_id": txn_id,
        "resolved_status": final_status,
        "action": action_taken,
        "retries_performed": retry_count,
        "processed_at": current_time
    }
'''

auth_code = '''"""
SentinelShop — Authentication & Session Management
"""
import hashlib
from typing import Dict, Any, List


def hash_api_key(api_key: str) -> str:
    """Deterministic SHA256 API key hash (Expected: LOW RISK)."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def sanitize_username(username: str) -> str:
    """Basic username sanitization (Expected: LOW RISK)."""
    return username.strip().lower()[:32]


def validate_session_and_roles(session_data: Dict[str, Any], required_roles: List[str]) -> bool:
    """Basic session authorization check (Expected: LOW RISK)."""
    if not session_data or not session_data.get("active"):
        return False
    user_roles = session_data.get("roles", [])
    for role in required_roles:
        if role in user_roles:
            return True
    return False
'''

(orders_dir / "checkout.py").write_text(checkout_code, encoding="utf-8")
(payments_dir / "gateway.py").write_text(gateway_code, encoding="utf-8")
(users_dir / "auth.py").write_text(auth_code, encoding="utf-8")

# Step 2: Git simulation
def run_git(args, env=None):
    current_env = os.environ.copy()
    if env:
        current_env.update(env)
    res = subprocess.run(
        ["git"] + args,
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        env=current_env
    )
    if res.returncode != 0:
        print(f"Git failed: {' '.join(args)} -> {res.stderr}")
    return res

run_git(["init"])
run_git(["config", "user.name", "Atharva Nanaware"])
run_git(["config", "user.email", "atharva@sentinel.org"])

# 16 distinct team contributors
authors = [
    ("Alice Dev", "alice@sentinel.org"),
    ("Bob Engineer", "bob@sentinel.org"),
    ("Charlie Architect", "charlie@sentinel.org"),
    ("Diana Security", "diana@sentinel.org"),
    ("Evan Lead", "evan@sentinel.org"),
    ("Fiona QA", "fiona@sentinel.org"),
    ("George SRE", "george@sentinel.org"),
    ("Hannah Backend", "hannah@sentinel.org"),
    ("Ian Core", "ian@sentinel.org"),
    ("Julia Platform", "julia@sentinel.org"),
    ("Kevin DevOps", "kevin@sentinel.org"),
    ("Laura Payments", "laura@sentinel.org"),
    ("Mike Fullstack", "mike@sentinel.org"),
    ("Nina Distributed", "nina@sentinel.org"),
    ("Oscar Release", "oscar@sentinel.org"),
    ("Atharva Nanaware", "atharva@sentinel.org")
]

# Genesis commit: 2025-01-10
start_date = datetime(2025, 1, 10, 9, 30, 0, tzinfo=timezone.utc)
run_git(["add", "."])
date_str = start_date.strftime("%Y-%m-%d %H:%M:%S +0000")
run_git(["commit", "-m", "feat: initial commit of SentinelShop microservice architecture"],
        env={"GIT_AUTHOR_NAME": authors[0][0], "GIT_AUTHOR_EMAIL": authors[0][1],
             "GIT_AUTHOR_DATE": date_str, "GIT_COMMITTER_DATE": date_str})

# 36 Historical Commits designed to produce:
# - checkout.py: 15 authors, 14 bug fixes, churn ~ 2200, recent churn ~ 1100 -> HIGH RISK for process_complex_checkout
# - gateway.py: 10 authors, 7 bug fixes, churn ~ 1200, recent churn ~ 500 -> MEDIUM RISK for reconcile_transaction_webhook
# - auth.py: 2 authors, 1 bug fix, churn ~ 30, recent churn 0 -> LOW RISK
commit_history = [
    # 2025 History (Older commits)
    ("checkout.py", "fix(checkout): fix initial coupon stacking calculation bug", 0, 30, 70),
    ("gateway.py",  "fix(payments): fix webhook HMAC signature mismatch error", 1, 55, 45),
    ("checkout.py", "refactor(checkout): optimize tier pricing matrix", 2, 75, 75),
    ("gateway.py",  "fix(payments): fix race condition in charge status updates", 3, 95, 45),
    ("checkout.py", "fix(checkout): patch float precision roundoff error in tax calculation", 4, 115, 80),
    ("auth.py",     "fix(auth): fix session invalidation on logout", 5, 130, 15),
    ("checkout.py", "feat(checkout): add express shipping logic and weight brackets", 6, 150, 75),
    ("gateway.py",  "patch(payments): handle gateway network socket timeout error", 7, 170, 50),
    ("checkout.py", "fix(checkout): resolve null reference bug when address missing", 8, 190, 80),
    ("gateway.py",  "refactor(payments): enhance timing attack prevention in HMAC digest", 9, 210, 55),
    ("checkout.py", "fix(checkout): fix boundary error when cart value equals zero", 10, 230, 85),
    ("gateway.py",  "fix(payments): patch partial refund ledger balance discrepancy", 11, 250, 50),
    ("checkout.py", "refactor(checkout): modularize currency multiplier table", 12, 275, 75),
    ("checkout.py", "fix(checkout): fix critical discount leakage on expired vouchers", 13, 300, 85),
    ("gateway.py",  "fix(payments): resolve deadlock in transaction state machine", 14, 325, 55),
    ("checkout.py", "patch(checkout): fix hazardous item international shipping restriction", 15, 350, 80),
    ("checkout.py", "fix(checkout): fix loyalty points overflow on enterprise tier", 0, 380, 80),
    ("gateway.py",  "fix(payments): fix dispute status webhook parsing error", 1, 410, 55),

    # 2026 Recent History (Within last 90 days: days ~ 600 to 680)
    ("checkout.py", "fix(checkout): patch emergency memory leak in currency converter", 2, 600, 95),
    ("gateway.py",  "fix(payments): resolve unhandled idempotency key duplicate exception", 3, 610, 60),
    ("checkout.py", "fix(checkout): fix tax jurisdiction rate for EU cross-border VAT", 4, 620, 100),
    ("gateway.py",  "refactor(payments): upgrade dead letter queue retry backoff", 5, 630, 60),
    ("checkout.py", "fix(checkout): resolve race condition in inventory stock reservation", 6, 640, 110),
    ("checkout.py", "patch(checkout): update fraud detection velocity risk threshold", 7, 648, 90),
    ("gateway.py",  "fix(payments): fix subscription renewal zero amount billing bug", 8, 655, 60),
    ("checkout.py", "fix(checkout): fix coupon code case sensitivity bug under load", 9, 662, 100),
    ("checkout.py", "refactor(checkout): streamline cart validation and exception routing", 10, 668, 105),
    ("gateway.py",  "patch(payments): sanitize transaction id special characters", 11, 672, 55),
    ("checkout.py", "fix(checkout): fix rounding anomaly in combined loyalty and promo rules", 12, 676, 105),
    ("gateway.py",  "fix(payments): resolve critical escrow hold bug on dispute rebuttal", 13, 678, 60),
    ("checkout.py", "refactor(checkout): production release readiness audit and cleanup", 14, 680, 95),
    ("checkout.py", "fix(checkout): patch critical vulnerability in high value order verification", 15, 681, 110),
]

for i, (target, msg, author_idx, days_offset, churn_lines) in enumerate(commit_history):
    commit_date = start_date + timedelta(days=days_offset)
    date_str = commit_date.strftime("%Y-%m-%d %H:%M:%S +0000")
    author = authors[author_idx % len(authors)]
    
    if target == "checkout.py":
        filepath = orders_dir / "checkout.py"
    elif target == "gateway.py":
        filepath = payments_dir / "gateway.py"
    else:
        filepath = users_dir / "auth.py"
        
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"\n# Patch {i+1} [{date_str[:10]}] by {author[0]}: {msg}\n")
        for line_num in range(churn_lines):
            f.write(f"# Internal assertion check {line_num} for audit verification rule {i}\n")
            
    run_git(["add", "."])
    run_git(["commit", "-m", msg],
            env={"GIT_AUTHOR_NAME": author[0], "GIT_AUTHOR_EMAIL": author[1],
                 "GIT_AUTHOR_DATE": date_str, "GIT_COMMITTER_DATE": date_str})

print(f"Reseeded SentinelShop with {len(commit_history) + 1} realistic commits across {len(authors)} developers.")
