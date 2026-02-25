"""
Module 1: data_simulator.py — CSAO Rail Recommendation System
==============================================================

Generates realistic synthetic data mimicking a Zomato-style food delivery platform.
Models city-wise behavior, meal composition logic, temporal patterns, user heterogeneity,
and real-world noise (abandoned carts, cold-start users, incomplete meals).

Generative Process:
    1. Sample user profile (city, veg_preference, price_sensitivity, segment)
    2. Sample session context (timestamp → meal_time, day_of_week)
    3. Sample restaurant given (city, cuisine_preference, meal_time)
    4. Generate cart via sequential meal-building with probabilistic item selection
    5. For each cart state transition, generate candidate add-on items with accept/reject labels
    6. Inject noise: abandoned carts, single-item sessions, random exploration

Author: CSAO MVP Sprint — Day 1
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import hashlib
import warnings

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# REASON: Seed for reproducibility across all random operations
RANDOM_SEED = 42

# REASON: These cities represent distinct food cultures in India,
# enabling city-wise behavioral differences in the simulation
CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata"]

# REASON: Meal time buckets align with real ordering peaks on food delivery platforms
MEAL_TIMES = {
    "breakfast": (6, 10),    # 6 AM – 10 AM
    "lunch":     (11, 14),   # 11 AM – 2 PM
    "snacks":    (15, 17),   # 3 PM – 5 PM
    "dinner":    (18, 22),   # 6 PM – 10 PM
    "late_night": (22, 2),   # 10 PM – 2 AM (wraps around midnight)
}

# REASON: User segments model the power-law distribution of ordering frequency
# observed in real platforms (many infrequent users, few power users)
USER_SEGMENTS = {
    "new":     {"order_range": (1, 3),   "fraction": 0.40},
    "regular": {"order_range": (4, 20),  "fraction": 0.45},
    "power":   {"order_range": (21, 60), "fraction": 0.15},
}

# REASON: Cuisine-city affinity matrix encodes regional food preferences.
# Higher values = stronger affinity. These drive restaurant/item selection.
CITY_CUISINE_AFFINITY = {
    "Mumbai":    {"North Indian": 0.7, "South Indian": 0.4, "Chinese": 0.6, "Street Food": 0.9, "Mughlai": 0.5, "Continental": 0.5, "Desserts": 0.6},
    "Delhi":     {"North Indian": 0.9, "South Indian": 0.3, "Chinese": 0.6, "Street Food": 0.8, "Mughlai": 0.9, "Continental": 0.4, "Desserts": 0.7},
    "Bangalore": {"North Indian": 0.5, "South Indian": 0.9, "Chinese": 0.5, "Street Food": 0.5, "Mughlai": 0.3, "Continental": 0.7, "Desserts": 0.5},
    "Hyderabad": {"North Indian": 0.5, "South Indian": 0.6, "Chinese": 0.4, "Street Food": 0.5, "Mughlai": 0.9, "Continental": 0.3, "Desserts": 0.6},
    "Chennai":   {"North Indian": 0.3, "South Indian": 0.95,"Chinese": 0.4, "Street Food": 0.6, "Mughlai": 0.2, "Continental": 0.4, "Desserts": 0.5},
    "Kolkata":   {"North Indian": 0.5, "South Indian": 0.3, "Chinese": 0.7, "Street Food": 0.8, "Mughlai": 0.6, "Continental": 0.4, "Desserts": 0.8},
}


# ============================================================================
# MENU CATALOG — Realistic item hierarchy
# ============================================================================

# REASON: Menu items are organized by (category, meal_role) to enable
# meal composition logic. Each item has attributes needed for feature engineering:
# cuisine, veg/non-veg, price tier, typical meal_times, and a natural language
# description (used later for LLM embedding in Module 3).

@dataclass
class MenuItem:
    """Represents a single item on the platform menu."""
    item_id: str
    name: str
    category: str          # e.g., "Main", "Side", "Dessert", "Drink", "Starter"
    cuisine: str           # e.g., "North Indian", "Mughlai"
    is_veg: bool
    price: float           # in INR
    meal_role: str         # semantic role: "anchor", "complement", "addon", "finisher"
    meal_times: List[str]  # when this item is typically ordered
    description: str       # natural language description for LLM embedding
    popularity_score: float = 0.5  # base popularity [0, 1]


def build_menu_catalog() -> List[MenuItem]:
    """
    Build a realistic menu catalog with ~80 items spanning multiple cuisines,
    categories, and meal roles. Items are designed to have natural pairing
    relationships (biryani→raita, dosa→chutney, etc.).

    Returns:
        List[MenuItem]: Full menu catalog.
    """
    catalog = [
        # ===================== NORTH INDIAN / MUGHLAI — MAINS (Anchors) =====================
        MenuItem("ITEM_001", "Chicken Biryani", "Main", "Mughlai", False, 299, "anchor",
                 ["lunch", "dinner"], "Fragrant basmati rice layered with spiced chicken, slow-cooked dum style"),
        MenuItem("ITEM_002", "Mutton Biryani", "Main", "Mughlai", False, 399, "anchor",
                 ["lunch", "dinner"], "Rich mutton biryani with saffron-infused rice and whole spices", 0.45),
        MenuItem("ITEM_003", "Veg Biryani", "Main", "Mughlai", True, 229, "anchor",
                 ["lunch", "dinner"], "Mixed vegetable biryani with paneer and aromatic spices", 0.55),
        MenuItem("ITEM_004", "Butter Chicken", "Main", "North Indian", False, 279, "anchor",
                 ["lunch", "dinner"], "Creamy tomato-based curry with tender tandoori chicken pieces"),
        MenuItem("ITEM_005", "Paneer Butter Masala", "Main", "North Indian", True, 249, "anchor",
                 ["lunch", "dinner"], "Rich and creamy paneer curry in buttery tomato gravy", 0.6),
        MenuItem("ITEM_006", "Dal Makhani", "Main", "North Indian", True, 199, "anchor",
                 ["lunch", "dinner"], "Slow-cooked black lentils in creamy buttery sauce", 0.65),
        MenuItem("ITEM_007", "Chicken Tikka Masala", "Main", "North Indian", False, 289, "anchor",
                 ["lunch", "dinner"], "Grilled chicken tikka in spiced onion-tomato masala"),
        MenuItem("ITEM_008", "Chole Bhature", "Main", "North Indian", True, 179, "anchor",
                 ["breakfast", "lunch"], "Spicy chickpea curry with deep-fried puffed bread", 0.6),

        # ===================== SOUTH INDIAN — MAINS (Anchors) =====================
        MenuItem("ITEM_009", "Masala Dosa", "Main", "South Indian", True, 149, "anchor",
                 ["breakfast", "lunch", "snacks"], "Crispy rice crepe filled with spiced potato masala", 0.7),
        MenuItem("ITEM_010", "Idli Sambar", "Main", "South Indian", True, 99, "anchor",
                 ["breakfast"], "Steamed rice cakes served with lentil sambar and chutneys", 0.65),
        MenuItem("ITEM_011", "Hyderabadi Dum Biryani", "Main", "Mughlai", False, 349, "anchor",
                 ["lunch", "dinner"], "Authentic Hyderabadi-style dum biryani with tender meat and fried onions", 0.55),
        MenuItem("ITEM_012", "Medu Vada", "Main", "South Indian", True, 89, "anchor",
                 ["breakfast", "snacks"], "Crispy lentil fritters served with sambar and coconut chutney", 0.5),
        MenuItem("ITEM_013", "Uttapam", "Main", "South Indian", True, 129, "anchor",
                 ["breakfast", "lunch"], "Thick rice pancake topped with onions, tomatoes, and chilies", 0.4),

        # ===================== CHINESE / INDO-CHINESE — MAINS =====================
        MenuItem("ITEM_014", "Chicken Fried Rice", "Main", "Chinese", False, 199, "anchor",
                 ["lunch", "dinner", "late_night"], "Wok-tossed rice with chicken, egg, and vegetables"),
        MenuItem("ITEM_015", "Veg Hakka Noodles", "Main", "Chinese", True, 169, "anchor",
                 ["lunch", "dinner", "late_night"], "Stir-fried noodles with mixed vegetables in soy sauce", 0.55),
        MenuItem("ITEM_016", "Chicken Manchurian", "Main", "Chinese", False, 219, "anchor",
                 ["lunch", "dinner"], "Deep-fried chicken in tangy Manchurian sauce with bell peppers"),
        MenuItem("ITEM_017", "Paneer Chilli", "Main", "Chinese", True, 209, "anchor",
                 ["lunch", "dinner"], "Crispy paneer tossed in spicy chilli garlic sauce", 0.5),

        # ===================== STREET FOOD — MAINS/SNACKS =====================
        MenuItem("ITEM_018", "Pav Bhaji", "Main", "Street Food", True, 149, "anchor",
                 ["lunch", "snacks", "dinner"], "Spiced mashed vegetable curry with buttered bread rolls", 0.65),
        MenuItem("ITEM_019", "Vada Pav", "Main", "Street Food", True, 59, "anchor",
                 ["breakfast", "lunch", "snacks"], "Mumbai's iconic spiced potato fritter in a bun", 0.7),
        MenuItem("ITEM_020", "Pani Puri", "Starter", "Street Food", True, 79, "addon",
                 ["snacks"], "Crispy hollow puris filled with spiced water and potato", 0.6),

        # ===================== CONTINENTAL =====================
        MenuItem("ITEM_021", "Margherita Pizza", "Main", "Continental", True, 249, "anchor",
                 ["lunch", "dinner", "late_night"], "Classic pizza with fresh mozzarella and basil on tomato sauce", 0.55),
        MenuItem("ITEM_022", "Chicken Burger", "Main", "Continental", False, 179, "anchor",
                 ["lunch", "dinner", "late_night"], "Grilled chicken patty with lettuce, cheese, and mayo in a sesame bun"),
        MenuItem("ITEM_023", "Veg Pasta", "Main", "Continental", True, 199, "anchor",
                 ["lunch", "dinner"], "Penne in creamy white sauce with mushrooms and broccoli", 0.45),

        # ===================== SIDES / COMPLEMENTS =====================
        MenuItem("ITEM_030", "Garlic Naan", "Side", "North Indian", True, 59, "complement",
                 ["lunch", "dinner"], "Soft tandoor-baked bread with roasted garlic butter", 0.75),
        MenuItem("ITEM_031", "Butter Naan", "Side", "North Indian", True, 49, "complement",
                 ["lunch", "dinner"], "Fluffy naan bread brushed with melted butter", 0.7),
        MenuItem("ITEM_032", "Tandoori Roti", "Side", "North Indian", True, 29, "complement",
                 ["lunch", "dinner"], "Whole wheat flatbread baked in clay tandoor oven", 0.6),
        MenuItem("ITEM_033", "Raita", "Side", "North Indian", True, 49, "complement",
                 ["lunch", "dinner"], "Cool yogurt with cucumber, onion, and cumin — perfect biryani companion", 0.65),
        MenuItem("ITEM_034", "Mirchi Ka Salan", "Side", "Mughlai", True, 99, "complement",
                 ["lunch", "dinner"], "Tangy peanut-sesame gravy with green chilies — classic biryani side", 0.5),
        MenuItem("ITEM_035", "Papad", "Side", "North Indian", True, 25, "addon",
                 ["lunch", "dinner"], "Crispy lentil wafer, roasted or fried", 0.6),
        MenuItem("ITEM_036", "Green Salad", "Side", "Continental", True, 69, "addon",
                 ["lunch", "dinner"], "Fresh cucumber, tomato, onion salad with lemon dressing", 0.35),
        MenuItem("ITEM_037", "French Fries", "Side", "Continental", True, 99, "complement",
                 ["lunch", "dinner", "snacks", "late_night"], "Golden crispy potato fries with seasoning", 0.65),
        MenuItem("ITEM_038", "Coconut Chutney", "Side", "South Indian", True, 29, "complement",
                 ["breakfast", "lunch", "snacks"], "Fresh ground coconut chutney with tempered mustard seeds", 0.55),
        MenuItem("ITEM_039", "Sambar", "Side", "South Indian", True, 49, "complement",
                 ["breakfast", "lunch"], "Tangy lentil stew with drumstick, tomato, and tamarind", 0.6),

        # ===================== STARTERS =====================
        MenuItem("ITEM_040", "Chicken Tikka", "Starter", "North Indian", False, 199, "addon",
                 ["lunch", "dinner"], "Smoky chargrilled chicken marinated in yogurt and spices", 0.55),
        MenuItem("ITEM_041", "Paneer Tikka", "Starter", "North Indian", True, 189, "addon",
                 ["lunch", "dinner"], "Grilled cottage cheese cubes with bell peppers in tandoori marinade", 0.5),
        MenuItem("ITEM_042", "Chicken Wings", "Starter", "Continental", False, 229, "addon",
                 ["dinner", "late_night"], "Crispy fried chicken wings tossed in hot sauce", 0.45),
        MenuItem("ITEM_043", "Spring Rolls", "Starter", "Chinese", True, 149, "addon",
                 ["lunch", "dinner", "snacks"], "Crispy rolls stuffed with mixed vegetables and glass noodles", 0.45),
        MenuItem("ITEM_044", "Manchow Soup", "Starter", "Chinese", True, 119, "addon",
                 ["lunch", "dinner"], "Spicy vegetable soup topped with crispy fried noodles", 0.4),
        MenuItem("ITEM_045", "Tomato Soup", "Starter", "Continental", True, 99, "addon",
                 ["lunch", "dinner"], "Classic creamy tomato soup with croutons and herbs", 0.35),

        # ===================== DESSERTS (Finishers) =====================
        MenuItem("ITEM_050", "Gulab Jamun", "Dessert", "North Indian", True, 89, "finisher",
                 ["lunch", "dinner"], "Soft milk-solid dumplings soaked in rose-cardamom syrup", 0.65),
        MenuItem("ITEM_051", "Rasmalai", "Dessert", "North Indian", True, 109, "finisher",
                 ["lunch", "dinner"], "Flattened paneer discs soaked in sweetened saffron milk", 0.5),
        MenuItem("ITEM_052", "Kheer", "Dessert", "North Indian", True, 79, "finisher",
                 ["lunch", "dinner"], "Creamy rice pudding with cardamom, nuts, and saffron", 0.45),
        MenuItem("ITEM_053", "Brownie with Ice Cream", "Dessert", "Continental", True, 149, "finisher",
                 ["lunch", "dinner", "late_night"], "Warm chocolate brownie topped with vanilla ice cream", 0.5),
        MenuItem("ITEM_054", "Kulfi", "Dessert", "North Indian", True, 69, "finisher",
                 ["lunch", "dinner", "snacks"], "Traditional Indian ice cream with pistachio and cardamom", 0.55),
        MenuItem("ITEM_055", "Payasam", "Dessert", "South Indian", True, 79, "finisher",
                 ["lunch", "dinner"], "South Indian milk pudding with vermicelli, cashews, and raisins", 0.4),

        # ===================== DRINKS (Finishers) =====================
        MenuItem("ITEM_060", "Masala Chai", "Drink", "North Indian", True, 39, "finisher",
                 ["breakfast", "snacks"], "Strong spiced tea with ginger and cardamom", 0.7),
        MenuItem("ITEM_061", "Cold Coffee", "Drink", "Continental", True, 99, "finisher",
                 ["lunch", "snacks", "late_night"], "Blended iced coffee with milk and chocolate drizzle", 0.5),
        MenuItem("ITEM_062", "Mango Lassi", "Drink", "North Indian", True, 79, "finisher",
                 ["lunch", "dinner"], "Creamy yogurt smoothie with Alphonso mango pulp", 0.55),
        MenuItem("ITEM_063", "Buttermilk (Chaas)", "Drink", "North Indian", True, 39, "finisher",
                 ["lunch", "dinner"], "Spiced salted buttermilk with cumin and mint — digestive cooler", 0.6),
        MenuItem("ITEM_064", "Thumbs Up / Cola", "Drink", "Continental", True, 40, "finisher",
                 ["lunch", "dinner", "late_night"], "Chilled cola — classic meal companion", 0.65),
        MenuItem("ITEM_065", "Fresh Lime Soda", "Drink", "Continental", True, 59, "finisher",
                 ["lunch", "dinner", "snacks"], "Refreshing lime soda, sweet or salted", 0.5),
        MenuItem("ITEM_066", "Filter Coffee", "Drink", "South Indian", True, 49, "finisher",
                 ["breakfast", "snacks"], "Traditional South Indian filter-drip coffee with frothy milk", 0.6),
        MenuItem("ITEM_067", "Jaljeera", "Drink", "North Indian", True, 49, "finisher",
                 ["lunch", "dinner"], "Tangy cumin-mint cooler — traditional Indian digestive drink", 0.4),
        MenuItem("ITEM_068", "Water Bottle", "Drink", "Continental", True, 20, "finisher",
                 ["breakfast", "lunch", "dinner", "snacks", "late_night"], "Packaged drinking water 500ml", 0.3),

        # ===================== EXTRAS / ADD-ONS =====================
        MenuItem("ITEM_070", "Extra Gravy", "Side", "North Indian", True, 49, "addon",
                 ["lunch", "dinner"], "Extra portion of rich curry gravy", 0.3),
        MenuItem("ITEM_071", "Egg (Boiled/Fried)", "Side", "Continental", False, 25, "addon",
                 ["breakfast", "lunch", "dinner"], "Boiled or fried egg — quick protein add-on", 0.35),
        MenuItem("ITEM_072", "Pickle (Achar)", "Side", "North Indian", True, 19, "addon",
                 ["lunch", "dinner"], "Spicy mango or mixed pickle", 0.3),
        MenuItem("ITEM_073", "Curd (Dahi)", "Side", "North Indian", True, 35, "addon",
                 ["lunch", "dinner"], "Fresh plain yogurt — cooling accompaniment for spicy meals", 0.45),
    ]
    return catalog


# ============================================================================
# MEAL COMPOSITION TEMPLATES — Encodes how real meals are built
# ============================================================================

# REASON: Meal templates define the "grammar" of a meal. Real users build meals
# in a structured way: anchor first, then complements, then optional dessert/drink.
# These templates drive the sequential cart-building logic.

MEAL_TEMPLATES = {
    "biryani_meal": {
        "anchors": ["ITEM_001", "ITEM_002", "ITEM_003", "ITEM_011"],
        "complements": ["ITEM_033", "ITEM_034", "ITEM_035"],       # raita, salan, papad
        "addons": ["ITEM_040", "ITEM_041", "ITEM_044"],            # starters
        "finishers": ["ITEM_050", "ITEM_051", "ITEM_064", "ITEM_063"],  # dessert + drink
        "completion_prob": {"complement": 0.65, "addon": 0.25, "finisher": 0.50},
    },
    "north_indian_curry": {
        "anchors": ["ITEM_004", "ITEM_005", "ITEM_006", "ITEM_007"],
        "complements": ["ITEM_030", "ITEM_031", "ITEM_032"],       # naans, roti
        "addons": ["ITEM_040", "ITEM_041", "ITEM_035", "ITEM_072"],
        "finishers": ["ITEM_050", "ITEM_052", "ITEM_062", "ITEM_064"],
        "completion_prob": {"complement": 0.80, "addon": 0.20, "finisher": 0.45},
    },
    "south_indian_breakfast": {
        "anchors": ["ITEM_009", "ITEM_010", "ITEM_012", "ITEM_013"],
        "complements": ["ITEM_038", "ITEM_039"],                    # chutney, sambar
        "addons": ["ITEM_073"],
        "finishers": ["ITEM_066", "ITEM_060"],                      # filter coffee, chai
        "completion_prob": {"complement": 0.70, "addon": 0.15, "finisher": 0.55},
    },
    "chinese_meal": {
        "anchors": ["ITEM_014", "ITEM_015", "ITEM_016", "ITEM_017"],
        "complements": ["ITEM_043", "ITEM_044"],                    # spring rolls, soup
        "addons": ["ITEM_037"],                                     # fries
        "finishers": ["ITEM_064", "ITEM_061", "ITEM_053"],
        "completion_prob": {"complement": 0.45, "addon": 0.30, "finisher": 0.40},
    },
    "street_food_snack": {
        "anchors": ["ITEM_018", "ITEM_019", "ITEM_020", "ITEM_008"],
        "complements": [],
        "addons": ["ITEM_037", "ITEM_035"],
        "finishers": ["ITEM_060", "ITEM_064", "ITEM_065"],
        "completion_prob": {"complement": 0.10, "addon": 0.20, "finisher": 0.50},
    },
    "continental_meal": {
        "anchors": ["ITEM_021", "ITEM_022", "ITEM_023"],
        "complements": ["ITEM_037", "ITEM_036"],                    # fries, salad
        "addons": ["ITEM_042", "ITEM_045"],                         # wings, soup
        "finishers": ["ITEM_053", "ITEM_061", "ITEM_064"],
        "completion_prob": {"complement": 0.55, "addon": 0.30, "finisher": 0.45},
    },
}

# REASON: Maps meal_time → which templates are likely. Breakfast doesn't have biryani.
MEALTIME_TEMPLATE_WEIGHTS = {
    "breakfast":   {"south_indian_breakfast": 0.5, "street_food_snack": 0.3, "continental_meal": 0.2},
    "lunch":       {"biryani_meal": 0.25, "north_indian_curry": 0.30, "chinese_meal": 0.15,
                    "south_indian_breakfast": 0.10, "continental_meal": 0.10, "street_food_snack": 0.10},
    "snacks":      {"street_food_snack": 0.5, "chinese_meal": 0.2, "continental_meal": 0.2, "south_indian_breakfast": 0.1},
    "dinner":      {"biryani_meal": 0.30, "north_indian_curry": 0.30, "chinese_meal": 0.20,
                    "continental_meal": 0.15, "street_food_snack": 0.05},
    "late_night":  {"chinese_meal": 0.35, "continental_meal": 0.35, "street_food_snack": 0.20,
                    "biryani_meal": 0.10},
}


# ============================================================================
# USER PROFILE GENERATOR
# ============================================================================

@dataclass
class UserProfile:
    """Simulated user with behavioral preferences."""
    user_id: str
    city: str
    segment: str           # "new", "regular", "power"
    veg_preference: float  # 0.0 = strictly non-veg, 1.0 = strictly veg
    price_sensitivity: float  # 0.0 = price-insensitive, 1.0 = very price-sensitive
    adventurousness: float    # 0.0 = orders same items, 1.0 = tries new things
    preferred_cuisines: List[str] = field(default_factory=list)
    num_orders: int = 0


def generate_users(n_users: int = 5000, rng: np.random.Generator = None) -> List[UserProfile]:
    """
    Generate diverse user profiles with city-correlated preferences.

    The veg_preference parameter is drawn from a Beta distribution whose
    shape parameters vary by city, reflecting real dietary patterns:
    - Chennai/Bangalore: higher veg tendency (α=3, β=2)
    - Delhi/Kolkata: mixed (α=2, β=2)
    - Mumbai/Hyderabad: slightly non-veg leaning (α=2, β=3)

    Args:
        n_users: Number of users to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        List[UserProfile]: Generated user profiles.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    # REASON: City-specific Beta distribution params for veg_preference
    # α > β → veg-leaning, α < β → non-veg-leaning
    city_veg_beta_params = {
        "Mumbai":    (2.0, 2.5),
        "Delhi":     (2.5, 2.5),
        "Bangalore": (3.0, 2.0),
        "Hyderabad": (1.8, 3.0),
        "Chennai":   (3.5, 2.0),
        "Kolkata":   (2.0, 2.8),
    }

    # REASON: City population weights for realistic user distribution
    city_weights = np.array([0.25, 0.25, 0.18, 0.15, 0.10, 0.07])
    city_weights /= city_weights.sum()

    users = []
    for i in range(n_users):
        city = rng.choice(CITIES, p=city_weights)
        alpha_v, beta_v = city_veg_beta_params[city]

        # Determine user segment based on configured fractions
        seg_roll = rng.random()
        cumulative = 0.0
        segment = "new"
        for seg_name, seg_info in USER_SEGMENTS.items():
            cumulative += seg_info["fraction"]
            if seg_roll <= cumulative:
                segment = seg_name
                break

        lo, hi = USER_SEGMENTS[segment]["order_range"]
        num_orders = rng.integers(lo, hi + 1)

        # REASON: Derive cuisine preferences from city affinity + personal noise
        city_affinities = CITY_CUISINE_AFFINITY[city]
        preferred = [c for c, score in city_affinities.items()
                     if rng.random() < score * 0.8]  # threshold at 80% of affinity
        if not preferred:
            preferred = [max(city_affinities, key=city_affinities.get)]

        user = UserProfile(
            user_id=f"U{i+1:05d}",
            city=city,
            segment=segment,
            veg_preference=float(rng.beta(alpha_v, beta_v)),
            price_sensitivity=float(rng.beta(2, 3)),       # most users are moderately sensitive
            adventurousness=float(rng.beta(2, 5)),          # most users are habitual
            preferred_cuisines=preferred,
            num_orders=num_orders,
        )
        users.append(user)

    return users


# ============================================================================
# SESSION GENERATOR — Sequential cart building with candidate labels
# ============================================================================

@dataclass
class SessionEvent:
    """A single cart state transition within a session, with candidate labels."""
    session_id: str
    user_id: str
    city: str
    timestamp: datetime
    meal_time: str
    day_of_week: int          # 0=Monday, 6=Sunday
    cart_items: List[str]     # item_ids currently in cart
    last_item_added: str      # the item that triggered this event
    candidate_item_id: str    # item being evaluated as add-on
    candidate_accepted: int   # 1 = user added it, 0 = did not
    cart_total_price: float
    cart_item_count: int
    cart_veg_ratio: float
    meal_template: str        # which meal template is driving this session
    is_abandoned: bool        # True if session was ultimately abandoned (no order placed)


def _get_meal_time(hour: int) -> str:
    """Map hour of day to meal time bucket."""
    if 6 <= hour < 10:
        return "breakfast"
    elif 10 <= hour < 15:       # REASON: extended to 3pm to cover late lunches
        return "lunch"
    elif 15 <= hour < 18:
        return "snacks"
    elif 18 <= hour < 22:
        return "dinner"
    else:
        return "late_night"


def _sample_timestamp(rng: np.random.Generator, base_date: datetime) -> datetime:
    """
    Sample a session timestamp with peak-hour bias.

    Uses a mixture of Gaussians centered on lunch (12:30) and dinner (19:30)
    peaks, plus a uniform background for off-peak orders.

    Args:
        rng: Random generator.
        base_date: The date for this session.

    Returns:
        datetime: Sampled timestamp.
    """
    # REASON: Real food delivery has bimodal peaks at lunch and dinner
    peak_choice = rng.random()
    if peak_choice < 0.35:
        # Lunch peak: N(12.5, 1.2)
        hour = rng.normal(12.5, 1.2)
    elif peak_choice < 0.70:
        # Dinner peak: N(19.5, 1.5)
        hour = rng.normal(19.5, 1.5)
    elif peak_choice < 0.82:
        # Breakfast: N(8.0, 0.8)
        hour = rng.normal(8.0, 0.8)
    elif peak_choice < 0.92:
        # Snacks: N(16.0, 0.8)
        hour = rng.normal(16.0, 0.8)
    else:
        # Late night: N(23.0, 1.0)
        hour = rng.normal(23.0, 1.0)

    hour = np.clip(hour, 0, 23.99)
    minutes = int((hour % 1) * 60)
    return base_date.replace(hour=int(hour), minute=minutes, second=0)


def generate_sessions(
    users: List[UserProfile],
    catalog: List[MenuItem],
    n_days: int = 30,
    start_date: datetime = datetime(2025, 1, 1),
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Generate realistic session data with sequential cart building.

    For each user, generates their configured number of orders spread across
    the simulation period. Each session follows a meal template with probabilistic
    item additions, generating (cart_state, candidate, accepted) tuples for training.

    Args:
        users: List of UserProfile objects.
        catalog: List of MenuItem objects.
        n_days: Number of days to simulate.
        start_date: Start date of simulation.
        rng: Random generator.

    Returns:
        pd.DataFrame: DataFrame of SessionEvent records.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    # Build lookup structures
    item_lookup: Dict[str, MenuItem] = {item.item_id: item for item in catalog}
    all_item_ids = list(item_lookup.keys())

    events: List[Dict] = []
    session_counter = 0

    for user in users:
        # Spread user's orders across the simulation period
        order_dates = sorted(rng.choice(n_days, size=user.num_orders, replace=True))

        for order_idx, day_offset in enumerate(order_dates):
            session_counter += 1
            session_id = f"S{session_counter:07d}"
            base_date = start_date + timedelta(days=int(day_offset))
            timestamp = _sample_timestamp(rng, base_date)
            meal_time = _get_meal_time(timestamp.hour)

            # REASON: Select meal template based on meal_time + city cuisine affinity
            template_weights = MEALTIME_TEMPLATE_WEIGHTS.get(meal_time, MEALTIME_TEMPLATE_WEIGHTS["lunch"])
            template_names = list(template_weights.keys())
            template_probs = np.array(list(template_weights.values()))

            # Adjust template probs by user's cuisine preferences
            for j, tname in enumerate(template_names):
                template = MEAL_TEMPLATES[tname]
                # Check if template's anchors align with user's cuisine preferences
                anchor_cuisines = set(item_lookup[aid].cuisine for aid in template["anchors"] if aid in item_lookup)
                overlap = len(anchor_cuisines.intersection(set(user.preferred_cuisines)))
                template_probs[j] *= (1.0 + 0.3 * overlap)

            template_probs /= template_probs.sum()
            chosen_template_name = rng.choice(template_names, p=template_probs)
            template = MEAL_TEMPLATES[chosen_template_name]

            # ---- Build cart sequentially ----
            cart: List[str] = []

            # Step 1: Select anchor item
            valid_anchors = [aid for aid in template["anchors"] if aid in item_lookup]
            if not valid_anchors:
                continue

            # REASON: Filter anchors by veg preference
            scored_anchors = []
            for aid in valid_anchors:
                item = item_lookup[aid]
                veg_match = 1.0 if (item.is_veg and user.veg_preference > 0.5) or \
                                   (not item.is_veg and user.veg_preference <= 0.5) else 0.3
                price_match = max(0.1, 1.0 - user.price_sensitivity * (item.price / 400.0))
                scored_anchors.append((aid, veg_match * price_match * item.popularity_score))

            anchor_ids, anchor_scores = zip(*scored_anchors)
            anchor_probs = np.array(anchor_scores, dtype=float)
            anchor_probs /= anchor_probs.sum()
            anchor_id = rng.choice(list(anchor_ids), p=anchor_probs)
            cart.append(anchor_id)

            # Step 2: Sequentially add complements, addons, finishers
            # At each step, generate candidate items and label them
            is_abandoned = rng.random() < 0.08  # REASON: ~8% cart abandonment rate

            for role in ["complement", "addon", "finisher"]:
                role_items = template.get(f"{role}s", template.get(role, []))
                if not role_items:
                    continue

                base_accept_prob = template["completion_prob"].get(role, 0.3)

                # REASON: Power users are more likely to build complete meals
                segment_multiplier = {"new": 0.7, "regular": 1.0, "power": 1.3}[user.segment]
                accept_prob = min(0.95, base_accept_prob * segment_multiplier)

                # REASON: Weekend orders tend to be larger (more add-ons)
                if timestamp.weekday() >= 5:
                    accept_prob = min(0.95, accept_prob * 1.15)

                # Generate candidates for this role
                valid_candidates = [cid for cid in role_items if cid in item_lookup and cid not in cart]

                # Also add some noise candidates (random items not in cart)
                noise_candidates = rng.choice(
                    [iid for iid in all_item_ids if iid not in cart and iid not in valid_candidates],
                    size=min(3, len(all_item_ids) - len(cart) - len(valid_candidates)),
                    replace=False
                ).tolist()

                all_candidates = valid_candidates + noise_candidates

                for cand_id in all_candidates:
                    cand_item = item_lookup[cand_id]

                    # Compute acceptance probability for this specific candidate
                    cand_accept = accept_prob

                    # Reduce prob for noise candidates (they don't fit the template)
                    if cand_id in noise_candidates:
                        cand_accept *= 0.08  # REASON: Random items rarely accepted

                    # Veg preference filter
                    if cand_item.is_veg and user.veg_preference < 0.3:
                        cand_accept *= 0.6
                    elif not cand_item.is_veg and user.veg_preference > 0.7:
                        cand_accept *= 0.05  # REASON: Strong veg users almost never accept non-veg

                    # Price sensitivity
                    if cand_item.price > 200 and user.price_sensitivity > 0.7:
                        cand_accept *= 0.4

                    # Meal time compatibility
                    if meal_time not in cand_item.meal_times:
                        cand_accept *= 0.15

                    accepted = 1 if rng.random() < cand_accept else 0

                    # Compute cart state features
                    cart_prices = sum(item_lookup[cid].price for cid in cart)
                    cart_veg_count = sum(1 for cid in cart if item_lookup[cid].is_veg)

                    event = {
                        "session_id": session_id,
                        "user_id": user.user_id,
                        "city": user.city,
                        "user_segment": user.segment,
                        "user_veg_preference": round(user.veg_preference, 3),
                        "user_price_sensitivity": round(user.price_sensitivity, 3),
                        "timestamp": timestamp,
                        "meal_time": meal_time,
                        "day_of_week": timestamp.weekday(),
                        "hour_of_day": timestamp.hour,
                        "is_weekend": int(timestamp.weekday() >= 5),
                        "cart_items": cart.copy(),
                        "cart_item_count": len(cart),
                        "cart_total_price": cart_prices,
                        "cart_veg_ratio": round(cart_veg_count / len(cart), 2) if cart else 0.0,
                        "last_item_added": cart[-1],
                        "last_item_category": item_lookup[cart[-1]].category,
                        "last_item_cuisine": item_lookup[cart[-1]].cuisine,
                        "cart_categories": list(set(item_lookup[cid].category for cid in cart)),
                        "cart_cuisines": list(set(item_lookup[cid].cuisine for cid in cart)),
                        "candidate_item_id": cand_id,
                        "candidate_category": cand_item.category,
                        "candidate_cuisine": cand_item.cuisine,
                        "candidate_price": cand_item.price,
                        "candidate_is_veg": int(cand_item.is_veg),
                        "candidate_meal_role": cand_item.meal_role,
                        "candidate_popularity": cand_item.popularity_score,
                        "candidate_accepted": accepted,
                        "meal_template": chosen_template_name,
                        "is_abandoned": int(is_abandoned),
                    }
                    events.append(event)

                    # If accepted, add to cart for next round
                    if accepted == 1:
                        cart.append(cand_id)

            # REASON: Also generate single-item sessions as noise (~5% of sessions)
            # These represent users who order quickly without browsing add-ons
            if rng.random() < 0.05 and len(events) > 0:
                events[-1]["is_abandoned"] = 1  # mark last event as from abandoned flow

    df = pd.DataFrame(events)
    return df


# ============================================================================
# DATA SUMMARY & VALIDATION
# ============================================================================

def validate_and_summarize(df: pd.DataFrame) -> Dict:
    """
    Run validation checks and produce summary statistics.

    Checks:
        - No null values in critical columns
        - Acceptance rate is within realistic bounds (5-40%)
        - City distribution matches expected weights
        - Temporal distribution shows peak-hour patterns

    Returns:
        Dict with summary statistics.
    """
    summary = {}

    # Basic shape
    summary["total_events"] = len(df)
    summary["unique_sessions"] = df["session_id"].nunique()
    summary["unique_users"] = df["user_id"].nunique()

    # BUSINESS LINK: Acceptance rate directly maps to CSAO rail click-through rate
    summary["overall_acceptance_rate"] = round(df["candidate_accepted"].mean(), 4)

    # Acceptance by segment
    summary["acceptance_by_segment"] = (
        df.groupby("user_segment")["candidate_accepted"].mean().round(4).to_dict()
    )

    # Acceptance by meal_time
    summary["acceptance_by_meal_time"] = (
        df.groupby("meal_time")["candidate_accepted"].mean().round(4).to_dict()
    )

    # City distribution
    summary["city_distribution"] = (
        df.groupby("city")["session_id"].nunique().to_dict()
    )

    # Cart size distribution
    summary["avg_cart_size_at_event"] = round(df["cart_item_count"].mean(), 2)

    # Temporal: events by hour
    summary["events_by_hour"] = (
        df.groupby("hour_of_day").size().to_dict()
    )

    # Abandonment rate
    summary["abandonment_rate"] = round(df["is_abandoned"].mean(), 4)

    # Validation checks
    critical_cols = ["session_id", "user_id", "candidate_item_id", "candidate_accepted"]
    null_counts = df[critical_cols].isnull().sum().to_dict()
    if any(v > 0 for v in null_counts.values()):
        warnings.warn(f"Null values found in critical columns: {null_counts}")
    summary["null_check_passed"] = all(v == 0 for v in null_counts.values())

    acceptance_rate = summary["overall_acceptance_rate"]
    summary["acceptance_rate_realistic"] = 0.03 < acceptance_rate < 0.50

    return summary


# ============================================================================
# MAIN — Demo execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CSAO Rail Recommendation System — Module 1: Data Simulator")
    print("=" * 70)

    # Step 1: Build menu catalog
    catalog = build_menu_catalog()
    print(f"\n✓ Menu catalog: {len(catalog)} items")
    print(f"  Categories: {set(item.category for item in catalog)}")
    print(f"  Cuisines:   {set(item.cuisine for item in catalog)}")
    print(f"  Veg items:  {sum(1 for i in catalog if i.is_veg)}")
    print(f"  Non-veg:    {sum(1 for i in catalog if not i.is_veg)}")

    # Step 2: Generate users
    rng = np.random.default_rng(RANDOM_SEED)
    users = generate_users(n_users=5000, rng=rng)
    print(f"\n✓ Generated {len(users)} users")
    for seg, info in USER_SEGMENTS.items():
        count = sum(1 for u in users if u.segment == seg)
        print(f"  {seg:>8s}: {count} users ({count/len(users)*100:.1f}%)")

    # Step 3: Generate session data
    print("\n⏳ Generating sessions (this may take a minute)...")
    df = generate_sessions(users, catalog, n_days=30, rng=rng)
    print(f"✓ Generated {len(df):,} candidate events across {df['session_id'].nunique():,} sessions")

    # Step 4: Validate and summarize
    summary = validate_and_summarize(df)
    print(f"\n{'─' * 50}")
    print("DATA SUMMARY")
    print(f"{'─' * 50}")
    print(f"  Total events:           {summary['total_events']:,}")
    print(f"  Unique sessions:        {summary['unique_sessions']:,}")
    print(f"  Unique users:           {summary['unique_users']:,}")
    print(f"  Overall acceptance rate: {summary['overall_acceptance_rate']:.2%}")
    print(f"  Avg cart size at event:  {summary['avg_cart_size_at_event']}")
    print(f"  Abandonment rate:        {summary['abandonment_rate']:.2%}")
    print(f"  Null check passed:       {summary['null_check_passed']}")
    print(f"  Acceptance rate realistic: {summary['acceptance_rate_realistic']}")

    print(f"\n  Acceptance by segment:")
    for seg, rate in summary["acceptance_by_segment"].items():
        print(f"    {seg:>8s}: {rate:.2%}")

    print(f"\n  Acceptance by meal time:")
    for mt, rate in summary["acceptance_by_meal_time"].items():
        print(f"    {mt:>10s}: {rate:.2%}")

    print(f"\n  Sessions by city:")
    for city, count in sorted(summary["city_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {city:>12s}: {count:,}")

    # Step 5: Save to CSV for downstream modules
    output_path = "csao_training_data.csv"
    # REASON: Convert list columns to string for CSV storage; will parse back in feature_engineering.py
    df_save = df.copy()
    df_save["cart_items"] = df_save["cart_items"].apply(str)
    df_save["cart_categories"] = df_save["cart_categories"].apply(str)
    df_save["cart_cuisines"] = df_save["cart_cuisines"].apply(str)
    df_save.to_csv(output_path, index=False)
    print(f"\n✓ Saved training data to {output_path}")
    print(f"  Shape: {df_save.shape}")
    print(f"  Columns: {list(df_save.columns)}")

    # Show sample rows
    print(f"\n{'─' * 50}")
    print("SAMPLE EVENTS (first 5)")
    print(f"{'─' * 50}")
    sample_cols = ["session_id", "user_id", "city", "meal_time", "cart_item_count",
                   "candidate_item_id", "candidate_category", "candidate_accepted"]
    print(df[sample_cols].head(5).to_string(index=False))
    print("\n✅ Module 1 complete. Ready for Module 2: feature_engineering.py")
