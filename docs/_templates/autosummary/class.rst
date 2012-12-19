
{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in all_attributes %}
      {%- if not item.startswith('_') %}
         {{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}


   {% block methods %}
   {% if methods %}
   .. rubric:: Methods Summary

   .. autosummary::
   {% for item in all_methods %}
      {%- if not item.startswith('_') or item in ['__call__'] %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}


   {% block methods_documentation %}
   {% if methods %}

   .. rubric:: Methods Documentation

   {% for item in all_methods %}
   {%- if not item.startswith('_') or item in ['__call__'] %}
   .. automethod:: {{ item }}
   {%- endif -%}
   {%- endfor %}

   {% endif %}
   {% endblock %}
