"""
CryptoVault Honeypot URL Configuration
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from graphene_django.views import GraphQLView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('vault.urls')),
    # VULN: GraphQL with introspection enabled, no auth
    path('graphql/', GraphQLView.as_view(graphiql=True)),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# VULN: Expose debug toolbar and error pages
if settings.DEBUG:
    urlpatterns += [
        path('__debug__/', include('django.contrib.admindocs.urls')),
    ]
