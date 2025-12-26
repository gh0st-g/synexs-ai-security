"""
GraphQL Schema for CryptoVault
VULN: Introspection enabled, no auth, injection vulnerabilities
"""
import graphene
from graphene_django import DjangoObjectType
from django.db import connection
from .models import User, Wallet, Comment


class UserType(DjangoObjectType):
    """User GraphQL type"""
    class Meta:
        model = User
        fields = (
            'id', 'username', 'email', 'wallet_address', 'wallet_seed',
            'balance_btc', 'balance_eth', 'balance_usdt', 'is_premium'
        )


class WalletType(DjangoObjectType):
    """Wallet transaction GraphQL type"""
    class Meta:
        model = Wallet
        fields = '__all__'


class CommentType(DjangoObjectType):
    """Comment GraphQL type"""
    class Meta:
        model = Comment
        fields = '__all__'


class Query(graphene.ObjectType):
    """
    GraphQL Queries
    VULN: No authentication, SQL injection in raw queries
    """
    all_users = graphene.List(UserType)
    user_by_username = graphene.Field(UserType, username=graphene.String())
    user_by_id = graphene.Field(UserType, id=graphene.Int())

    all_wallets = graphene.List(WalletType)
    wallets_by_user = graphene.List(WalletType, user_id=graphene.Int())

    all_comments = graphene.List(CommentType)

    # VULN: Raw SQL query with injection
    search_users = graphene.List(UserType, query=graphene.String())

    def resolve_all_users(self, info):
        """Get all users - VULN: No auth required"""
        return User.objects.all()

    def resolve_user_by_username(self, info, username):
        """Get user by username"""
        try:
            return User.objects.get(username=username)
        except User.DoesNotExist:
            return None

    def resolve_user_by_id(self, info, id):
        """Get user by ID - VULN: IDOR"""
        try:
            return User.objects.get(id=id)
        except User.DoesNotExist:
            return None

    def resolve_all_wallets(self, info):
        """Get all wallets - VULN: No auth, exposes all transactions"""
        return Wallet.objects.all()

    def resolve_wallets_by_user(self, info, user_id):
        """Get wallets by user - VULN: IDOR"""
        return Wallet.objects.filter(user_id=user_id)

    def resolve_all_comments(self, info):
        """Get all comments"""
        return Comment.objects.all()

    def resolve_search_users(self, info, query):
        """
        Search users with raw SQL
        VULN: SQL Injection vulnerability
        """
        # VULN: Unsanitized SQL query
        sql = f"SELECT * FROM vault_users WHERE username LIKE '%{query}%' OR email LIKE '%{query}%'"

        try:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                columns = [col[0] for col in cursor.description]
                results = cursor.fetchall()

                # Convert to User objects (simplified)
                user_ids = [row[0] for row in results]
                return User.objects.filter(id__in=user_ids)
        except Exception:
            return []


class CreateUser(graphene.Mutation):
    """
    Create user mutation
    VULN: No input validation
    """
    class Arguments:
        username = graphene.String(required=True)
        email = graphene.String(required=True)
        password = graphene.String(required=True)

    user = graphene.Field(UserType)
    success = graphene.Boolean()

    def mutate(self, info, username, email, password):
        """VULN: No validation, weak password allowed"""
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )
        return CreateUser(user=user, success=True)


class Mutation(graphene.ObjectType):
    """GraphQL Mutations"""
    create_user = CreateUser.Field()


# Schema with introspection enabled (VULN)
schema = graphene.Schema(query=Query, mutation=Mutation)
