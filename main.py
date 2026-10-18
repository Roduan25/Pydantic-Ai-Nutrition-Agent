from dotenv import load_dotenv
from pydantic import BaseModel
import json
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

load_dotenv()

NUTRITION_DB = {
    "chicken breast": {"calories": 165, "protein_g": 31, "carbs_g": 0,  "fat_g": 3.6},
    "brown rice":     {"calories": 216, "protein_g": 5,  "carbs_g": 45, "fat_g": 1.8},
    "broccoli":       {"calories": 55,  "protein_g": 3.7,"carbs_g": 11, "fat_g": 0.6},
    "olive oil":      {"calories": 119, "protein_g": 0,  "carbs_g": 0,  "fat_g": 13.5},
}

@dataclass
class NutritionService:
    database: dict

    def lookup(self, ingredient: str) -> dict | None:
        return self.database.get(ingredient.lower().strip())

    def all_ingredients(self) -> list[str]:
        return list(self.database.keys())

class MealSummary(BaseModel):
    total_calories: int
    total_protein_g: float
    total_carbs_g: float
    total_fat_g: float
    health_verdict: str
    recommendation: str


agent = Agent(
    "openai:gpt-5.4-mini",
    output_type=MealSummary,
    deps_type=NutritionService,
    instructions="""
    استخدم الأدوات للبحث عن بيانات كل مكون غذائي.
    احسب المجموع التقريبي للسعرات والبروتين والكربوهيدرات والدهون بناءً على الكميات التي يذكرها المستخدم.
    أعطِ تقييمًا عامًا مختصرًا للوجبة وتوصية عملية.
    لا تقدم تشخيصًا طبيًا أو خطة علاجية.
    """,
)

# @agent.tool_plain
# def get_ingredient_nutrition(ingredient: str) -> str:
#     """
#     Look up calories, protein, carbs, and fat per 100g for a single ingredient.
#     Returns an error message if the ingredient is not found in the database.
#     """
#     data = NUTRITION_DB.get(ingredient.lower().strip())

#     if data:
#         return json.dumps({"ingredient": ingredient, **data})

#     return f"Not found. Available: {', '.join(NUTRITION_DB)}"


@agent.tool
def get_ingredient_nutrition(
    ctx: RunContext[NutritionService],
    ingredient: str,
) -> str:
    """
    Look up calories, protein, carbs, and fat per 100g for a single ingredient.
    """
    data = ctx.deps.lookup(ingredient)

    if data:
        return json.dumps({"ingredient": ingredient, **data})

    return f"Not found. Available: {', '.join(ctx.deps.all_ingredients())}"

service = NutritionService(database=NUTRITION_DB)



result = agent.run_sync("Analyse: 150g chicken breast, 200g brown rice.", deps=service)

print(result.output.model_dump())
