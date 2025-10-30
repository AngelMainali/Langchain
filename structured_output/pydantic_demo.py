from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'angel'  # angel is set as default value
    age: Optional[int] = None
    email : EmailStr
    cgpa: float = Field(gt=0, lt=4, default =2, description ='A number in float according to marks obtained')

new_student = { 'name':'ram', 'age':23, 'email':'abc@gmail.com','cgpa':3}

student = Student(**new_student)
student_dict = dict(student)

print(student_dict)
print(student_dict['age'])
print(type(student))